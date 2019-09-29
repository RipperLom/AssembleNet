#!/usr/bin/env python
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


class BlockGRU(object):
    """
        GRU BLOCK
    """

    def __init__(self, config_dir: str):
        '''
        config_dir : 该block 相关的conf
        '''
        self.DTYPE = 'float32'
        self.DTYPE_INT = 'int64'
        self.name = "ELMO_GRU_block"
        self._n_tokens_vocab = 0
        self._max_batch_size = 128

        try:
            with open(config_dir, 'r') as fin:
                self.options = json.load(fin)
        except:
            self.options = {}
        else:
            self.bidirectional = self.options['bidirectional']
            self.projection_dim = self.options['gru']['projection_dim']
            self.gru_dim = self.options['gru']['dim']
            self.n_gru_layers = self.options['gru'].get('n_layers', 2)
            self.use_skip_connections = self.options['gru']['use_skip_connections']
            self.unroll_steps = self.options['unroll_steps']
            self.batch_size = self.options['batch_size']
            self.keep_prob = 1.0 - self.options['dropout']

        self.load_state = False
        self.weight_file = ''
        self.embedding_weight_file = ''
        # for each direction, we'll store tensors for each layer
        self.gru_outputs = {'forward': [], 'backward': []}
        self.gru_state_sizes = {'forward': [], 'backward': []}
        self.gru_init_states = {'forward': [], 'backward': []}
        self.gru_final_states = {'forward': [], 'backward': []}
        self.init_gru_state = []
        self.final_gru_state = []
    
    
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
            outfile: path of a HDF5 file
        Returns:
            state of dump from ckpt to hdf5, bool in [True, False]
        Raises:
            None
        '''
        config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=config) as sess:
            with tf.variable_scope('lm'):
                # we use the "Saver" class to load the variables
                loader = tf.train.import_meta_graph(ckpt_file + '.meta')
                loader.restore(sess, ckpt_file)

            with h5py.File(outfile, 'w') as fout:
                for v in tf.trainable_variables():
                    if v.name.find('softmax') >= 0:
                        # don't dump these
                        continue
                    
                    outname = self._get_outname(v.name)
                    print("Saving variable {0} with name {1}".format(
                        v.name, outname))
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
        initialize ELMO with embedding and gru cells
        Args:
            input_tensor: tensor of batch ids, shape = [batchSize, seqLen]
            input_tensor_reverse: reverse tensor of batch ids, shape = [batchSize, seqLen]
            embedding_weights: init embedding table from gensim.word2vec
        Returns:
            output_tensor: tensor of word representation, shape = [batchSize, seqLen, hiddenDim]
        Raises:
            None
        '''
        # get the gru inputs
        self.embedding = tf.nn.embedding_lookup(
            embedding_weights, input_tensor)
        self.embedding_reverse = tf.nn.embedding_lookup(
            embedding_weights, input_tensor_reverse)
        gru_inputs = [self.embedding, self.embedding_reverse]

        # build gru outputs
        return self._build_grus_outputs(gru_inputs)
    
    
    def _build_grus_outputs(self, gru_inputs):
        '''
        build grus outputs
        Args:
            gru_inputs: grus inputs
        Returns:
            gru_outputs: grus outputs
        Raises:
            None
        '''
        gru_outputs = []
        for gru_num, gru_input in enumerate(gru_inputs):
            # build gru cell
            gru_cells = []
            for i in range(self.n_gru_layers):
                gru_cell = tf.nn.rnn_cell.GRUCell(self.gru_dim)

                if self.use_skip_connections:
                    # ResidualWrapper adds inputs to outputs
                    if i == 0:
                        # don't add skip connection from token embedding to 1st layer output
                        pass
                    else:
                        # add a skip connection
                        gru_cell = tf.nn.rnn_cell.ResidualWrapper(gru_cell)

                # add dropout for train

                gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell,
                                                          input_keep_prob=self.keep_prob)

                gru_cells.append(gru_cell)

            if self.n_gru_layers > 1:
                gru_cell = tf.nn.rnn_cell.MultiRNNCell(gru_cells)
            else:
                gru_cell = gru_cells[0]

            with tf.control_dependencies([gru_input]):
                self.init_gru_state.append(
                    gru_cell.zero_state(self.batch_size, self.DTYPE))
                # NOTE: this variable scope is for backward compatibility
                # with existing models...
                with tf.variable_scope('RNN_%s' % gru_num):
                    _gru_output_unpacked, final_state = tf.nn.static_rnn(
                        gru_cell,
                        tf.unstack(gru_input, axis=1),
                        initial_state=self.init_gru_state[-1])

                self.final_gru_state.append(final_state)

            # (self.batch_size * self.unroll_steps, 512)
            gru_output_flat = tf.reshape(
                tf.stack(_gru_output_unpacked, axis=1), [-1, self.gru_dim])

            # add dropout to output
            gru_output_flat = tf.nn.dropout(gru_output_flat, self.keep_prob)
            tf.add_to_collection('gru_output_embeddings', _gru_output_unpacked)
            gru_outputs.append(gru_output_flat)
        return gru_outputs


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
            outname = re.sub('/gru_cell/', '/multi_rnn_cell/cell_0/gru_cell/', tf_name)
        else:
            outname = tf_name
        outname = re.sub(':0$', '', outname)
        outname = outname.lstrip('lm/')
        outname = re.sub('/rnn/', '/RNN/', outname)
        outname = re.sub('/multi_rnn_cell/', '/MultiRNNCell/', outname)
        outname = re.sub('/cell_', '/Cell', outname)
        outname = re.sub('/gru_cell/', '/GRUCell/', outname)

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
        embedding and grus
        Args:
            input_tensor: tensor of batch ids, shape = [batchSize, seqLen]
        Returns:
            embedding block and gru block
        Raises:
            None
        '''
        self._build_embedding(input_tensor)
        self._build_grus()


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


    def _build_grus(self):
        """
        the GRUs --- these will collect the initial states for the forward
        and reverse GRUs
        Args:
            None
        Returns:
            gru block
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

            for i in range(self.n_gru_layers):

                if self.projection_dim < self.gru_dim:
                    # are projecting down output
                    gru_cell = tf.nn.rnn_cell.GRUCell(self.projection_dim)
                else:
                    gru_cell = tf.nn.rnn_cell.GRUCell(self.gru_dim)

                if self.use_skip_connections:
                    # ResidualWrapper adds inputs to outputs
                    if i == 0:
                        # don't add skip connection from token embedding to
                        # 1st layer output
                        pass
                    else:
                        # add a skip connection
                        gru_cell = tf.nn.rnn_cell.ResidualWrapper(gru_cell)

                # collect the input state, run the dynamic rnn, collect
                # the output
                state_size = gru_cell.state_size
                # the GRUs are stateful.  To support multiple batch sizes,
                # we'll allocate size for states up to max_batch_size,
                # then use the first batch_size entries for each batch
                init_states = [
                    tf.Variable(tf.zeros([self._max_batch_size, dim]),
                                trainable=False)
                    for dim in gru_cell.state_size
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
                        gru_cell,
                        layer_input,
                        sequence_length=self.sequence_lengths,
                        initial_state=tf.nn.rnn_cell.GRUStateTuple(
                            *batch_init_states),
                    )

                self.gru_state_sizes[direction].append(gru_cell.state_size)
                self.gru_init_states[direction].append(init_states)
                self.gru_final_states[direction].append(final_state)

                if direction == 'forward':
                    self.gru_outputs[direction].append(layer_output)
                else:
                    self.gru_outputs[direction].append(
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

            n_lm_layers = len(self.gru_outputs['forward'])
            for i in range(n_lm_layers):
                layers.append(
                    tf.concat(
                        [self.gru_outputs['forward'][i],
                         self.gru_outputs['backward'][i]],
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
                weight_name_map[root + '/rnn/gru_cell/kernel'] = \
                    root + '/GRUCell/W_0'
                weight_name_map[root + '/rnn/gru_cell/bias'] = \
                    root + '/GRUCell/B'
                weight_name_map[root + '/rnn/gru_cell/projection/kernel'] = \
                    root + '/GRUCell/W_P_0'

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

