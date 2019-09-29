#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/8/29 6:14 PM
# software: PyCharm

import numpy as np
import tensorflow as tf


class PairwiseHingeLoss(object):
    """
    a layer class: pairwise hinge loss
    """

    def __init__(self, config):
        """
        init function
        """
        self.margin = float(config["margin"])

    def ops(self, score_pos, score_neg):
        """
        operation
        """
        return tf.reduce_mean(tf.maximum(0., score_neg +
                                         self.margin - score_pos))


class PairwiseLogLoss(object):
    """
    a layer class: pairwise log loss
    """

    def __init__(self, config=None):
        """
        init function
        """
        pass

    def ops(self, score_pos, score_neg):
        """
        operation
        """
        return tf.reduce_mean(tf.nn.sigmoid(score_neg - score_pos))


class SoftmaxWithLoss(object):
    """
    a layer class: softmax loss
    """

    def __init__(self, config=None):
        """
        init function
        """
        pass

    def ops(self, pred, label):
        """
        operation
        """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,
                                                                      labels=label))


class BiSampledSoftmaxLoss(object):
    """
        a layer class: sampled softmax loss
    """

    def __init__(self, config=None):
        """
        init function
        """
        self.DTYPE = 'float32'
        self.n_negative_samples_batch = config['n_negative_samples_batch']
        self.n_tokens_vocab = config['n_tokens_vocab']
        self.projection_dim = config['dim']

        with tf.variable_scope('softmax'), tf.device('/cpu:0'):
            # Glorit init (std=(1.0 / sqrt(fan_in))
            softmax_init = tf.random_normal_initializer(0.0,
                            1.0 / np.sqrt(self.projection_dim))

            self.softmax_W = tf.get_variable(
                'W', [self.n_tokens_vocab, self.projection_dim],
                dtype=self.DTYPE, initializer=softmax_init
            )
            self.softmax_b = tf.get_variable(
                'b', [self.n_tokens_vocab],
                dtype=self.DTYPE, initializer=tf.constant_initializer(0.0))

        pass

    def ops(self, input_tensors, next_ids):
        '''
        an op to calculate losses
        loss for each direction of the LSTM
        Args:
            input_tensors: outputs of elmo embedding
            next_ids = [self.next_token_id, self.next_token_id_reverse]
        Returns:
            average loss of both directions
        Raises:
            None
        '''
        individual_losses = []

        for id_placeholder, lstm_output_flat in zip(next_ids, input_tensors):
            # flatten the LSTM output and next token id gold to shape:
            # (batch_size * unroll_steps, softmax_dim)
            # Flatten and reshape the token_id placeholders
            next_token_id_flat = tf.reshape(id_placeholder, [-1, 1])

            with tf.control_dependencies([lstm_output_flat]):
                losses = tf.nn.sampled_softmax_loss(
                    self.softmax_W, self.softmax_b,
                    next_token_id_flat, lstm_output_flat,
                    self.n_negative_samples_batch,
                    self.n_tokens_vocab,
                    num_true=1)

            individual_losses.append(tf.reduce_mean(losses))

        # now make the total loss -- it's the mean of the individual losses
        total_loss = 0.5 * (individual_losses[0] + individual_losses[1])

        return total_loss

