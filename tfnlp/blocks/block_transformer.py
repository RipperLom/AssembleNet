#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################

"""
File: block_transformer.py
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

class BlockTransformer(object):
    """
        Transformer BLOCK
    """
    def __init__(self, config = {}):
        '''
            config : 该block 相关的conf
        '''
        self.name = "transformer_block"
        self.hidden_size = config.get('hidden_size', 768)
        self.num_hidden_layers = config.get('num_hidden_layers', 3)
        self.num_attention_heads = config.get('num_attention_heads', 12)
        self.intermediate_size = config.get('intermediate_size', 768)
        self.hidden_act = config.get('hidden_act', 'gelu')
        self.hidden_dropout_prob = config.get('hidden_dropout_prob', 0.5)
        self.attention_probs_dropout_prob = config.get('attention_probs_dropout_prob', 0.5)
        self.initializer_range = config.get('initializer_range', 0.1)


        self.finish_load = False

    def load(self, weight_file):
        #读取h5py  1、先初始化config 2、init weight
        #
        self.finish_load = True
        return True

    def dump(self, weight_file):
        return True
    
    def ops(self, input_ids, input_mask, embedding_output):
        '''

        :param input_tensor:
        :return:
        '''
        if self.finish_load:
            pass
        else:
            attention_mask = create_attention_mask_from_input_mask(
                input_ids, input_mask)
            self.all_encoder_layers = transformer_model(
                input_tensor=embedding_output,
                attention_mask=attention_mask,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                intermediate_act_fn=self.get_activation(self.hidden_act),
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                initializer_range=self.initializer_range,
                do_return_all_layers=True)

            sequence_output = self.all_encoder_layers[-1]
            return sequence_output

    def get_activation(self, activation_string):
        """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

        Args:
          activation_string: String name of the activation function.

        Returns:
          A Python function corresponding to the activation function. If
          `activation_string` is None, empty, or "linear", this will return None.
          If `activation_string` is not a string, it will return `activation_string`.

        Raises:
          ValueError: The `activation_string` does not correspond to a known
            activation.
        """

        act = activation_string.lower()
        if act == "linear":
            return None
        elif act == "relu":
            return tf.nn.relu
        elif act == "gelu":
            return self.gelu
        elif act == "tanh":
            return tf.tanh
        else:
            raise ValueError("Unsupported activation: %s" % act)

    def gelu(self, input_tensor):
        """Gaussian Error Linear Unit.

        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415

        Args:
          input_tensor: float Tensor to perform activation.

        Returns:
          `input_tensor` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
        return input_tensor * cdf


