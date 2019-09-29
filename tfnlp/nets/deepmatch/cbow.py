#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: cbow.py
Author: wangyan 
Date: 2019/04/11 11:53:24
Brief: 
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import six
import json
import copy
import logging

import tensorflow as tf

from tfnlp import layers
from tfnlp import blocks


class CBOWConfig(object):
    def __init__(self):
        pass
    
    @classmethod
    def from_dict(cls, json_object):
        """
            load args form Python dictionary
        """
        config = CBOWConfig()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config
    
    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, json_file) as infile:
            text = infile.read()
            return cls.from_dict(json.loads(text))
    
    def to_dict(self):
        result = copy.deepcopy(self.__dict__)
        return result
    
    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class CBOWModel(object):
    """
    pre-trained elmo to do sentence embedding and match their results
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
            self.option_file = config['option_file']
            self.weight_file = config['weight_file']
            self.embedding_weight_file = config['embedding_weight_file']
            self.elmo_lstm_block = blocks.BlockLSTM(self.option_file)
            self.elmo_lstm_block.load(self.weight_file, self.embedding_weight_file)
            self.attention_pooling_layer = layers.AttentionPoolingLayer()
            self.deep_match_block = blocks.BlockDeepMatch()

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
        '''
        an op to predict the result before calculating the loss.
        Args:
            left_slots: ids of word in a batch of sentence
            right_slots: ids of word in a batch of sentence
        Returns:
            pred:
            （1）listwise：softmax result of multi-classification
            （2）pointwise：softmax result of binary classification
            （3）pairwise：cosine similarities
        Raises:
            None
        '''
        # classification
        if self.task_mode == 'listwise':
            left = left_slots[self.left_name]
            lstm_represent = self.elmo_lstm_block.ops(left)
            pooling = self.attention_pooling_layer.ops(lstm_represent)
            pooling_relu = tf.nn.relu(pooling)
            fc1 = self.fc1_layer.ops(pooling_relu)
            fc1_relu = tf.nn.relu(fc1)
            pred = self.fc2_layer.ops(fc1_relu)

        elif self.task_mode == 'pointwise':
            # left
            left = left_slots[self.left_name]
            left_represent = self.elmo_lstm_block.ops(left)
            left_pooling = self.attention_pooling_layer.ops(left_represent)
            # right
            right = right_slots[self.right_name]
            right_represent = self.elmo_lstm_block.ops(right)
            right_pooling = self.attention_pooling_layer.ops(right_represent)
            # cos
            pred = self.deep_match_block.ops(left_pooling, right_pooling)

        elif self.task_mode == 'pairwise':
            # neuron network
            left = left_slots[self.left_name]
            left_emb = self.emb_layer.ops(left)
            left_cnn = self.cnn_layer.ops(left_emb)
            left_relu = self.relu_layer.ops(left_cnn)
            right = right_slots[self.right_name]
            right_emb = self.emb_layer.ops(right)
            right_cnn = self.cnn_layer.ops(right_emb)
            right_relu = self.relu_layer.ops(right_cnn)
            hid1_left = self.fc1_layer.ops(left_relu)
            hid1_right = self.fc1_layer.ops(right_relu)
            left_relu2 = self.relu_layer.ops(hid1_left)
            right_relu2 = self.relu_layer.ops(hid1_right)
            pred = self.cos_layer.ops(left_relu2, right_relu2)

        else:
            print('error false task_mode')
            pred = None

        return pred

