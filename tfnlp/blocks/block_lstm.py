# !/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
#
########################################################################

"""
File: block_gru.py
Author: wangyan
Date: 2019/09/10 11:53:24
Brief:  block_gru
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import time
import json
import logging

import h5py
import numpy as np
import tensorflow as tf


class BlockLSTM(object):
    """
        LSTM BLOCK
    """
    
    def __init__(self, config_dir: str):
        '''
        config_dir : 该block 相关的conf
        '''
        self.DTYPE = 'float32'
        self.DTYPE_INT = 'int64'
        self.name = "ELMO_LSTM_block"
        self._n_tokens_vocab = 0
        self._max_batch_size = 128

        try:
            with open(config_dir, 'r') as fin:
                self.options = json.load(fin)
        except:
            self.options = {}
        else:
            self.bidirectional = self.options['bidirectional']
            self.projection_dim = self.options['lstm']['projection_dim']
            self.lstm_dim = self.options['lstm']['dim']
            self.n_lstm_layers = self.options['lstm'].get('n_layers', 2)
            self.cell_clip = self.options['lstm'].get('cell_clip')
            self.proj_clip = self.options['lstm'].get('proj_clip')
            self.use_skip_connections = self.options['lstm']['use_skip_connections']
            self.unroll_steps = self.options['unroll_steps']
            self.batch_size = self.options['batch_size']
            self.keep_prob = 1.0 - self.options['dropout']

        self.load_state = False
        self.weight_file = ''
        self.embedding_weight_file = ''
        # for each direction, we'll store tensors for each layer
        self.lstm_outputs = {'forward': [], 'backward': []}
        self.lstm_state_sizes = {'forward': [], 'backward': []}
        self.lstm_init_states = {'forward': [], 'backward': []}
        self.lstm_final_states = {'forward': [], 'backward': []}
        self.init_lstm_state = []
        self.final_lstm_state = []


    def load(self, weight_file, embedding_weight_file):
        '''
        check the existence of  pre-trained data
        Args:
            weight_file: path of lm_weights.hdf5
            emb_file: path of vocab_embedding.hdf5
        Returns:
            state of load hdf5 files, bool in [True, False]
        Raises:
            None
        '''
        # check option (config)
        if not self.options:
            print('option does not exist.')
            return False

        # check weight_file
        if not os.path.exists(weight_file):
            print('weight_file does not exist.')
            return False
        else:
            self.weight_file = weight_file

        # check emb_file
        if not os.path.exists(embedding_weight_file):
            print('emb_file does not exist.')
            return False
        else:
            self.embedding_weight_file = embedding_weight_file
            with h5py.File(self.embedding_weight_file, 'r') as fin:
                # +1 for padding
                self._n_tokens_vocab = fin['embedding'].shape[0] + 1

        self.load_state = True
        return self.load_state


    def dump(self, ckpt_file, outfile):
        '''
        Dump the trained weights saved by tf.train.Saver to a HDF5 file.
        Args:
            ckpt_file: path of checkpoint
            outfile: a HDF5 file of weight
            outfile_embed_weight: a HDF5 file of embed weight
        Returns:
            state of dump from ckpt to hdf5, bool in [True, False]
        Raises:
            None
        '''

        # emb_file = self.tf_save_dir + "/vocab_embedding.hdf5"
        # embed = sess.run(model.embedding_weights)
        # with h5py.File(emb_file, "w") as fout:
        #     fout.create_dataset("embedding", embed.shape, dtype='float32', data=embed)


        config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=config) as sess:
            with tf.variable_scope('lm'):
                # we use the "Saver" class to load the variables
                loader = tf.train.import_meta_graph(ckpt_file + '.meta')
                loader.restore(sess, ckpt_file)

            with h5py.File(outfile, 'w') as fout:
                variables = tf.trainable_variables()
                for variable in variables:
                    print(variable)

                for v in tf.trainable_variables():

                    if v.name.find('softmax') >= 0:
                        # don't dump these
                        continue
                    outname = self._get_outname(v.name)
                    print("Saving variable {0} with name {1}"
                        .format(v.name, outname))
                    shape = v.get_shape().as_list()
                    dset = fout.create_dataset(outname, shape, dtype='float32')
                    values = sess.run([v])[0]
                    dset[...] = values

        return True


    def ops(self, input_tensor, input_tensor_reverse = None, embedding_weights = None):
        '''
        an op to compute ELMo (weighted average of the internal biLM layers)
        Args:
            input_tensor: tensor of batch ids, shape = [batchSize, seqLen]
            input_tensor_reverse: reverse tensor of batch ids, shape = [batchSize, seqLen]
            embedding_weights: init embedding table from gensim.word2vec
        Returns:
            output_tensor: tensor of word representation, shape = [batchSize, seqLen, hiddenDim]
        Raises:
            None
        '''
        # 已经加载h5数据
        if self.load_state:
            return self._reload(input_tensor)

        else: # 初始化参数
            return self._first_load(input_tensor, input_tensor_reverse, embedding_weights)


    def _first_load(self, input_tensor, input_tensor_reverse, embedding_weights):
        '''
        initialize ELMO with embedding and lstm cells
        Args:
            input_tensor: tensor of batch ids, shape = [batchSize, seqLen]
            input_tensor_reverse: reverse tensor of batch ids, shape = [batchSize, seqLen]
            embedding_weights: init embedding table from gensim.word2vec
        Returns:
            output_tensor: tensor of word representation, shape = [batchSize, seqLen, hiddenDim]
        Raises:
            None
        '''
        # get the LSTM inputs
        self.embedding = tf.nn.embedding_lookup(
            embedding_weights, input_tensor)

        if self.bidirectional:
            self.embedding_reverse = tf.nn.embedding_lookup(
                embedding_weights, input_tensor_reverse)
            lstm_inputs = [self.embedding, self.embedding_reverse]
        else:
            lstm_inputs = [self.embedding]

        # build lstm outputs
        return self._build_lstms_outputs(lstm_inputs)


    def _build_lstms_outputs(self, lstm_inputs):
        '''
        build lstms outputs
        Args:
            lstm_inputs: lstms inputs
        Returns:
            lstm_outputs: lstms outputs
        Raises:
            None
        '''
        lstm_outputs = []
        for lstm_num, lstm_input in enumerate(lstm_inputs):
            # build lstm cell
            lstm_cells = []
            for i in range(self.n_lstm_layers):
                lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_dim,
                        cell_clip=self.cell_clip, proj_clip=self.proj_clip)

                if self.use_skip_connections:
                    # ResidualWrapper adds inputs to outputs
                    if i == 0:
                        # don't add skip connection from token embedding to 1st layer output
                        pass
                    else:
                        # add a skip connection
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                # add dropout for train

                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                          input_keep_prob=self.keep_prob)

                lstm_cells.append(lstm_cell)

            if self.n_lstm_layers > 1:
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
            else:
                lstm_cell = lstm_cells[0]

            with tf.control_dependencies([lstm_input]):
                self.init_lstm_state.append(
                    lstm_cell.zero_state(self.batch_size, self.DTYPE))
                # NOTE: this variable scope is for backward compatibility
                # with existing models...
                if self.bidirectional:
                    with tf.variable_scope('RNN_%s' % lstm_num):
                        _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                            lstm_cell,
                            tf.unstack(lstm_input, axis=1),
                            initial_state=self.init_lstm_state[-1])
                else:
                    _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                        lstm_cell,
                        tf.unstack(lstm_input, axis=1),
                        initial_state=self.init_lstm_state[-1])

                self.final_lstm_state.append(final_state)

            # (self.batch_size * self.unroll_steps, 512)
            lstm_output_flat = tf.reshape(
                tf.stack(_lstm_output_unpacked, axis=1), [-1, self.lstm_dim])

            # add dropout to output
            lstm_output_flat = tf.nn.dropout(lstm_output_flat, self.keep_prob)
            tf.add_to_collection('lstm_output_embeddings', _lstm_output_unpacked)
            lstm_outputs.append(lstm_output_flat)
        return lstm_outputs


    def _reload(self, input_tensor):
        '''
        with reloading pretrained hdf5 to initialize neuron network,
        we can switch word ids to word representation
        Args:
            input_tensor: tensor of batch ids,
                          shape = [batchSize, seqLen]
        Returns:
            output_tensor: tensor of word representation,
                           shape = [batchSize, seqLen, hiddenDim]
        Raises:
            None
        '''
        # the sequence lengths from input mask
        self.mask = input_tensor > 0
        self.sequence_lengths = tf.reduce_sum(tf.cast(self.mask, tf.int32), axis=1)
        self.batch_size = tf.shape(self.sequence_lengths)[0]

        with tf.variable_scope('bilm', custom_getter=self._custom_getter, reuse=tf.AUTO_REUSE):
            self._build_model(input_tensor)

        all_lm_embeddings = self._build_ops()
        # get all embedding layers
        lm_embeddings = all_lm_embeddings['lm_embeddings']
        n_lm_layers = int(lm_embeddings.get_shape()[1])
        layers = tf.split(lm_embeddings, n_lm_layers, axis=1)
        # just the top layer
        output_tensor = tf.squeeze(layers[-1], squeeze_dims=1)
        return output_tensor


    def _get_outname(self, tf_name):
        '''
        get outname from tf_name
        Args:
            tf_name: str
        Returns:
            out_name: str
        Raises:
            None
        '''
        if tf_name.find('multi_rnn_cell') == -1:
            outname = re.sub('/lstm_cell/', '/multi_rnn_cell/cell_0/lstm_cell/', tf_name)
        else:
            outname = tf_name
        outname = re.sub(':0$', '', outname)
        outname = outname.lstrip('lm/')
        outname = re.sub('/rnn/', '/RNN/', outname)
        outname = re.sub('/multi_rnn_cell/', '/MultiRNNCell/', outname)
        outname = re.sub('/cell_', '/Cell', outname)
        outname = re.sub('/lstm_cell/', '/LSTMCell/', outname)

        if '/RNN/' in outname:
            if 'projection' in outname:
                outname = re.sub('projection/kernel', 'W_P_0', outname)
            else:
                outname = re.sub('/kernel', '/W_0', outname)
                outname = re.sub('/bias', '/B', outname)

        return outname


    def _custom_getter(self, getter, name, *args, **kwargs):
        '''
        overwrite initializer
        Args:
            getter:
            name:
        Returns:
            values: pre-trained data in type --- np.ndarray
        Raises:
            None
        '''
        kwargs['trainable'] = True
        kwargs['initializer'] = self._pretrained_initializer(
            name, self.weight_file, self.embedding_weight_file
        )
        return getter(name, *args, **kwargs)


    def _build_model(self, input_tensor):
        '''
        embedding and lstms
        Args:
            input_tensor: tensor of batch ids, shape = [batchSize, seqLen]
        Returns:
            embedding block and lstm block
        Raises:
            None
        '''
        self._build_embedding(input_tensor)
        self._build_lstms()


    def _build_embedding(self, input_tensor):
        '''
        using vocab_embedding.hdf5 embedding_table to do embedding_lookup
        Args:
            input_tensor: tensor of batch ids, shape = [batchSize, seqLen]
        Returns:
            embedding block
        Raises:
            None
        '''
        # the word embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                "embedding", [self._n_tokens_vocab, self.projection_dim],
                dtype=self.DTYPE
            )
            self.embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    input_tensor)


    def _build_lstms(self):
        """
        the LSTMs --- these will collect the initial states for the forward
        and reverse LSTMs
        Args:
            None
        Returns:
            lstm block
        Raises:
            None
        """
        update_ops = []
        for direction in ['forward', 'backward']:

            if direction == 'forward':
                layer_input = self.embedding
            else:
                layer_input = tf.reverse_sequence(
                    self.embedding,
                    self.sequence_lengths,
                    seq_axis=1,
                    batch_axis=0)

            for i in range(self.n_lstm_layers):

                if self.projection_dim < self.lstm_dim:
                    # are projecting down output
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        self.lstm_dim, num_proj=self.projection_dim,
                        cell_clip=self.cell_clip, proj_clip=self.proj_clip)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        self.lstm_dim,
                        cell_clip=self.cell_clip, proj_clip=self.proj_clip)

                if self.use_skip_connections:
                    # ResidualWrapper adds inputs to outputs
                    if i == 0:
                        # don't add skip connection from token embedding to
                        # 1st layer output
                        pass
                    else:
                        # add a skip connection
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                # collect the input state, run the dynamic rnn, collect
                # the output
                state_size = lstm_cell.state_size
                # the LSTMs are stateful.  To support multiple batch sizes,
                # we'll allocate size for states up to max_batch_size,
                # then use the first batch_size entries for each batch
                init_states = [
                    tf.Variable(tf.zeros([self._max_batch_size, dim]),
                                trainable=False)
                    for dim in lstm_cell.state_size
                ]
                batch_init_states = [
                    state[:self.batch_size, :] for state in init_states
                ]

                if direction == 'forward':
                    i_direction = 0
                else:
                    i_direction = 1

                variable_scope_name = 'RNN_{0}/RNN/MultiRNNCell/Cell{1}'.format(i_direction, i)
                with tf.variable_scope(variable_scope_name):
                    layer_output, final_state = tf.nn.dynamic_rnn(
                        lstm_cell,
                        layer_input,
                        sequence_length=self.sequence_lengths,
                        initial_state=tf.nn.rnn_cell.LSTMStateTuple(
                            *batch_init_states),
                    )

                self.lstm_state_sizes[direction].append(lstm_cell.state_size)
                self.lstm_init_states[direction].append(init_states)
                self.lstm_final_states[direction].append(final_state)

                if direction == 'forward':
                    self.lstm_outputs[direction].append(layer_output)
                else:
                    self.lstm_outputs[direction].append(
                        tf.reverse_sequence(
                            layer_output,
                            self.sequence_lengths,
                            seq_axis=1,
                            batch_axis=0
                        )
                    )

                with tf.control_dependencies([layer_output]):
                    # update the initial states
                    for i in range(2):
                        new_state = tf.concat(
                            [final_state[i][:self.batch_size, :],
                             init_states[i][self.batch_size:, :]], axis=0)
                        state_update_op = tf.assign(init_states[i], new_state)
                        update_ops.append(state_update_op)

                layer_input = layer_output

        self.update_state_op = tf.group(*update_ops)


    def _build_ops(self):
        """
        build each-layer word representation in ELMO
        Args:
            None
        Returns:
            dict of
                'lm_embeddings': lm_embeddings,
                'lengths': sequence_length_wo_bos_eos,
                'token_embeddings': self.embedding,
                'mask': mask_wo_bos_eos,
        Raises:
            None
        """
        with tf.control_dependencies([self.update_state_op]):
            # get the LM embeddings
            token_embeddings = self.embedding
            layers = [
                tf.concat([token_embeddings, token_embeddings], axis=2)
            ]

            n_lm_layers = len(self.lstm_outputs['forward'])
            for i in range(n_lm_layers):
                layers.append(
                    tf.concat(
                        [self.lstm_outputs['forward'][i],
                         self.lstm_outputs['backward'][i]],
                        axis=-1
                    )
                )

            # The layers include the BOS/EOS tokens.  Remove them
            # sequence_length_wo_bos_eos = self.sequence_lengths - 2
            sequence_length_wo_bos_eos = self.sequence_lengths
            layers_without_bos_eos = []
            for layer in layers:
                layer_wo_bos_eos = layer[:, :, :]
                layer_wo_bos_eos = tf.reverse_sequence(
                    layer_wo_bos_eos,
                    self.sequence_lengths,
                    seq_axis=1,
                    batch_axis=0,
                )
                layer_wo_bos_eos = layer_wo_bos_eos[:, :, :]
                layer_wo_bos_eos = tf.reverse_sequence(
                    layer_wo_bos_eos,
                    sequence_length_wo_bos_eos,
                    seq_axis=1,
                    batch_axis=0,
                )
                layers_without_bos_eos.append(layer_wo_bos_eos)

            # concatenate the layers
            lm_embeddings = tf.concat(
                [tf.expand_dims(t, axis=1) for t in layers_without_bos_eos],
                axis=1
            )

            # get the mask op without bos/eos.
            # tf doesn't support reversing boolean tensors, so cast
            # to int then back
            mask_wo_bos_eos = tf.cast(self.mask[:, 1:], 'int32')
            mask_wo_bos_eos = tf.reverse_sequence(
                mask_wo_bos_eos,
                self.sequence_lengths,
                seq_axis=1,
                batch_axis=0,
            )
            mask_wo_bos_eos = mask_wo_bos_eos[:, 1:]
            mask_wo_bos_eos = tf.reverse_sequence(
                mask_wo_bos_eos,
                sequence_length_wo_bos_eos,
                seq_axis=1,
                batch_axis=0,
            )
            mask_wo_bos_eos = tf.cast(mask_wo_bos_eos, 'bool')

        return {
            'lm_embeddings': lm_embeddings,
            'lengths': sequence_length_wo_bos_eos,
            'token_embeddings': self.embedding,
            'mask': mask_wo_bos_eos,
        }


    def _pretrained_initializer(self, varname, weight_file, embedding_weight_file=None):
        '''
        We'll stub out all the initializers in the pretrained LM with
        a function that loads the weights from the file
        Args:
            varname: the tensor u want to reload from hdf5
            weight_file: path of lm_weights.hdf5
            embedding_weight_file: path of vocab_embedding.hdf5
        Returns:
            ret
        Raises:
            None
        '''
        weight_name_map = {}
        for i in range(2):
            for j in range(8):  # if we decide to add more layers
                root = 'RNN_{}/RNN/MultiRNNCell/Cell{}'.format(i, j)
                weight_name_map[root + '/rnn/lstm_cell/kernel'] = \
                    root + '/LSTMCell/W_0'
                weight_name_map[root + '/rnn/lstm_cell/bias'] = \
                    root + '/LSTMCell/B'
                weight_name_map[root + '/rnn/lstm_cell/projection/kernel'] = \
                    root + '/LSTMCell/W_P_0'

        # convert the graph name to that in the checkpoint
        varname_in_file = varname[5:]
        if varname_in_file.startswith('RNN'):
            varname_in_file = weight_name_map[varname_in_file]

        if varname_in_file == 'embedding':
            with h5py.File(embedding_weight_file, 'r') as fin:
                # Have added a special 0 index for padding not present
                # in the original model.
                embed_weights = fin[varname_in_file][...]
                weights = np.zeros(
                    (embed_weights.shape[0] + 1, embed_weights.shape[1]),
                    dtype=self.DTYPE
                )
                weights[1:, :] = embed_weights
        else:
            with h5py.File(weight_file, 'r') as fin:
                if varname_in_file == 'char_embed':
                    # Have added a special 0 index for padding not present
                    # in the original model.
                    char_embed_weights = fin[varname_in_file][...]
                    weights = np.zeros(
                        (char_embed_weights.shape[0] + 1,
                         char_embed_weights.shape[1]),
                        dtype=self.DTYPE
                    )
                    weights[1:, :] = char_embed_weights
                else:
                    weights = fin[varname_in_file][...]

        # Tensorflow initializers are callables that accept a shape parameter
        # and some optional kwargs
        def ret(shape, **kwargs):
            if list(shape) != list(weights.shape):
                raise ValueError(
                    "Invalid shape initializing {0}, got {1}, expected {2}".format(
                        varname_in_file, shape, weights.shape)
                )
            return weights

        return ret

