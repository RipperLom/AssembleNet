#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/9/24 10:09 AM
# software: PyCharm

import os
import time

import copy
import h5py
import json
import numpy as np
import tensorflow as tf

from tfnlp.nets.elmo.bilm.bilm_core import average_gradients, clip_grads, summary_gradient_updates
from tfnlp.nets.elmo.pretrained import Elmo2layerLstm
from tfnlp.nets.elmo.bilm.bilm_core import _get_feed_dict_from_X
from tfnlp.nets.elmo.bilm import bilm_util



class MultiGpuTrainer(object):
    '''
        多卡训练，feed_dict
    '''
    def __init__(self, config, dataset, net, loss, optimizer):
        self.config = config
        self.dataset = dataset
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

        self.n_gpus = config['n_gpus']
        # define the options
        self.tf_save_dir = self.config['save_dir']
        self.tf_log_dir = self.config['save_dir']
        self.restart_ckpt_file = self.config['restart_ckpt_file']


    def train(self, config, options):

        with tf.device('/cpu:0'):
            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0), trainable=False)

            # set up the optimizer
            lr = config.get('learning_rate', 0.2)
            opt = self.optimizer.ops()

            # calculate the gradients on each GPU
            models = []
            tower_grads = []
            train_perplexity = tf.get_variable('train_perplexity', [], initializer=tf.constant_initializer(0.0),
                                               trainable=False)

            norm_summaries = []
            for k in range(self.n_gpus):
                with tf.device('/gpu:%d' % k):
                    with tf.variable_scope('lm', reuse=k > 0):
                        # calculate the loss for one model replica and get lstm states
                        model = Elmo2layerLstm(config)
                        lstm_outputs, next_ids = model.ops()
                        loss = self.loss.ops(lstm_outputs, next_ids)
                        # loss = model.total_loss
                        models.append(model)

                        # get gradients
                        grads = opt.compute_gradients(
                            loss * options['unroll_steps'],
                            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                            )
                        tower_grads.append(grads)
                        # keep track of loss across all GPUs
                        train_perplexity += loss

            bilm_util.print_variable_summary()

            # calculate the mean of each gradient across all GPUs
            grads = average_gradients(tower_grads, options['batch_size'], options)
            grads, norm_summary_ops = clip_grads(grads, options, True, global_step)
            norm_summaries.extend(norm_summary_ops)

            # log the training perplexity
            train_perplexity = tf.exp(train_perplexity / self.n_gpus)
            perplexity_summmary = tf.summary.scalar('train_perplexity', train_perplexity)

            # some histogram summaries.  all models use the same parameters
            # so only need to summarize one
            histogram_summaries = [
                tf.summary.histogram('token_embedding', models[0].embedding)
            ]

            # tensors of the output from the LSTM layer
            lstm_out = tf.get_collection('lstm_output_embeddings')
            histogram_summaries.append(tf.summary.histogram('lstm_embedding_0', lstm_out[0]))
            if options.get('bidirectional', False):
                # also have the backward embedding
                histogram_summaries.append(
                    tf.summary.histogram('lstm_embedding_1', lstm_out[1]))

            # apply the gradients to create the training operation
            train_op = opt.apply_gradients(grads, global_step=global_step)

            # histograms of variables
            for v in tf.global_variables():
                histogram_summaries.append(tf.summary.histogram(v.name.replace(":", "_"), v))

            # get the gradient updates -- these aren't histograms, but we'll
            # only update them when histograms are computed
            histogram_summaries.extend(summary_gradient_updates(grads, opt, lr))

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
            summary_op = tf.summary.merge([perplexity_summmary] + norm_summaries)
            hist_summary_op = tf.summary.merge(histogram_summaries)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

        # do the training loop
        bidirectional = options.get('bidirectional', False)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init_op)

            # load the checkpoint data if needed
            if self.restart_ckpt_file is not None:
                loader = tf.train.Saver()
                loader.restore(sess, self.restart_ckpt_file)

            summary_writer = tf.summary.FileWriter(self.tf_log_dir, sess.graph)

            # For each batch:
            # Get a batch of data from the generator. The generator will
            # yield batches of size batch_size * self.n_gpus that are sliced
            # and fed for each required placeholer.
            #
            # We also need to be careful with the LSTM states.  We will
            # collect the final LSTM states after each batch, then feed
            # them back in as the initial state for the next batch
            batch_size = options['batch_size']
            unroll_steps = options['unroll_steps']
            n_train_tokens = options.get('n_train_tokens', 768648884)
            n_tokens_per_batch = batch_size * unroll_steps * self.n_gpus
            n_batches_per_epoch = int(n_train_tokens / n_tokens_per_batch)
            n_batches_total = options['n_epochs'] * n_batches_per_epoch

            print("Training for %s epochs and %s batches" % (options['n_epochs'], n_batches_total))

            # get the initial lstm states
            init_state_tensors = []
            final_state_tensors = []
            for model in models:
                init_state_tensors.extend(model.init_lstm_state)
                final_state_tensors.extend(model.final_lstm_state)


            feed_dict = {
                model.token_ids:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
                for model in models
            }

            feed_dict.update({
                model.token_ids_reverse:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
                for model in models
            })


            init_state_values = sess.run(init_state_tensors, feed_dict=feed_dict)

            t1 = time.time()
            data_gen = self.dataset.ops(batch_size * self.n_gpus, unroll_steps)
            for batch_no, batch in enumerate(data_gen, start=1):
                str_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print("==>[%s] training batch no %d ......" % (str_time, batch_no))

                # slice the input in the batch for the feed_dict
                X = batch
                feed_dict = {t: v for t, v in zip(
                    init_state_tensors, init_state_values)}
                for k in range(self.n_gpus):
                    model = models[k]
                    start = k * batch_size
                    end = (k + 1) * batch_size

                    char_inputs = None
                    feed_dict.update(
                        _get_feed_dict_from_X(X, start, end, model,
                                              char_inputs, bidirectional)
                    )

                # This runs the train_op, summaries and the "final_state_tensors"
                #   which just returns the tensors, passing in the initial
                #   state tensors, token ids and next token ids
                if batch_no % 1250 != 0:
                    ret = sess.run(
                        [train_op, summary_op, train_perplexity] +
                        final_state_tensors,
                        feed_dict=feed_dict
                    )

                    # first three entries of ret are:
                    #  train_op, summary_op, train_perplexity
                    # last entries are the final states -- set them to
                    # init_state_values
                    # for next batch
                    init_state_values = ret[3:]

                else:
                    # also run the histogram summaries
                    ret = sess.run(
                        [train_op, summary_op, train_perplexity, hist_summary_op] +
                        final_state_tensors,
                        feed_dict=feed_dict
                    )
                    init_state_values = ret[4:]

                if batch_no % 1250 == 0:
                    summary_writer.add_summary(ret[3], batch_no)

                if batch_no % 100 == 0:
                    # write the summaries to tensorboard and display perplexity
                    summary_writer.add_summary(ret[1], batch_no)
                    print("Batch %s, train_perplexity=%s  total_time=%0.4f s" % \
                          (batch_no, ret[2], time.time() - t1))

                if (batch_no % 1250 == 0) or (batch_no == n_batches_total):
                    # save the model
                    checkpoint_path = os.path.join(self.tf_save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)

                if batch_no == n_batches_total:
                    # done training!
                    emb_file = self.tf_save_dir + "/vocab_embedding.hdf5"
                    embed = sess.run(model.embedding_weights)
                    with h5py.File(emb_file, "w") as fout:
                        fout.create_dataset("embedding", embed.shape, dtype='float32', data=embed)
                    break

    def predict(self, config):
        pass

