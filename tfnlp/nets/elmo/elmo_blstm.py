#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/9/16 4:47 PM
# software: PyCharm

import logging

from tfnlp import layers
from tfnlp import blocks
import tensorflow as tf

class LstmElmo(object):
    """
    mlp cnn init function
    """

    def __init__(self, config):
        # 超参
        self.vocab_size = int(config['vocabulary_size'])
        self.emb_size = int(config['embedding_dim'])
        self.kernel_size = int(config['num_filters'])
        self.win_size = int(config['window_size'])
        self.hidden_size = int(config['hidden_size'])
        self.left_name, self.seq_len = config['left_slots'][0]
        self.right_name, self.seq_len = config['right_slots'][0]
        self.task_mode = config['training_mode']

        # neuron network
        self.emb_layer = layers.EmbeddingLayer(self.vocab_size, self.emb_size)
        self.cnn_layer = layers.CNNLayer(self.seq_len, self.emb_size, self.win_size, self.kernel_size)
        self.relu_layer = layers.ReluLayer()
        self.concat_layer = layers.ConcatLayer()

        # classification
        if self.task_mode == "pointwise":
            self.n_class = int(config['n_class'])
            self.fc1_layer = layers.FCLayer(2 * self.kernel_size, self.hidden_size)
        elif self.task_mode == "pairwise":
            self.fc1_layer = layers.FCLayer(self.kernel_size, self.hidden_size)
            self.cos_layer = layers.CosineLayer()
        elif self.task_mode == "listwise":
            self.n_class = int(config['n_class'])
            self.option_file = config['option_file']
            self.weight_file = config['weight_file']
            self.embedding_weight_file = config['embedding_weight_file']
            self.elmo_lstm_block = blocks.BlockLSTM(self.option_file)
            self.elmo_lstm_block.load(self.weight_file, self.embedding_weight_file)
            self.attention_pooling_layer = layers.AttentionPoolingLayer()
            self.fc1_layer = layers.FCLayer(2 * self.kernel_size, self.hidden_size)
            self.fc2_layer = layers.FCLayer(self.hidden_size, self.n_class)
        else:
            logging.error("training mode not supported")

    def predict(self, left_slots, right_slots):
        """
        predict graph of this net
        """
        # classification
        if self.task_mode == 'listwise':
            left = left_slots[self.left_name]
            lstm_represent = self.elmo_lstm_block.ops(left)
            pooling = self.attention_pooling_layer.ops(lstm_represent)
            pooling_relu = tf.nn.relu(pooling)
            fc1 = self.fc1_layer.ops(pooling_relu)
            fc1_relu = tf.nn.relu(fc1)
            pred = self.fc2_layer.ops(fc1_relu)
        else:
            # neuron network
            left = left_slots[self.left_name]
            left_emb = self.emb_layer.ops(left)
            left_cnn = self.cnn_layer.ops(left_emb)
            left_relu = self.relu_layer.ops(left_cnn)
            right = right_slots[self.right_name]
            right_emb = self.emb_layer.ops(right)
            right_cnn = self.cnn_layer.ops(right_emb)
            right_relu = self.relu_layer.ops(right_cnn)
            if self.task_mode == "pointwise":
                concat = self.concat_layer.ops([left_relu, right_relu], self.kernel_size * 2)
                concat_fc = self.fc1_layer.ops(concat)
                concat_relu = self.relu_layer.ops(concat_fc)
                pred = self.fc2_layer.ops(concat_relu)
            elif self.task_mode == "pairwise":
                hid1_left = self.fc1_layer.ops(left_relu)
                hid1_right = self.fc1_layer.ops(right_relu)
                left_relu2 = self.relu_layer.ops(hid1_left)
                right_relu2 = self.relu_layer.ops(hid1_right)
                pred = self.cos_layer.ops(left_relu2, right_relu2)

        return pred

