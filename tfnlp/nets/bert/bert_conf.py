#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: bert_conf.py
Author: wangyan 
Date: 2019/09/10 11:53:24
Brief: bert config
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import re
import six
import json
import copy


class BertConfig(object):
    def __init__(self, 
                vocab_size = 0,
                hidden_size = 768,
                num_hidden_layers = 12,
                num_attention_heads = 12,
                intermediate_size = 3072,
                hidden_act = "gelu",
                hidden_dropout_prob = 0.1,
                attention_probs_dropout_prob = 0.1,
                max_position_embeddings = 512,
                type_vocab_size = 16,
                initializer_range = 0.02):
        """
        Args:
            vocab_size:  Vocabulary size of `inputs_ids`
            hidden_size: Size of the encoder layers and the pooler layer.

            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in the Transformer encoder.
            
            intermediate_size: The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            hidden_act:  encoder 和池化的激活函数
            hidden_dropout_prob: The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size:   The vocabulary size of the `token_type_ids` passed into `BertModel`.
            initializer_range: The stdev of the truncated_normal_initializer for initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


    @classmethod
    def from_dict(cls, json_object):
        """
            load args form Python dictionary
        """
        config = BertConfig()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config
    
    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r") as infile:
            text = infile.read()
            return cls.from_dict(json.loads(text))
    
    def to_dict(self):
        result = copy.deepcopy(self.__dict__)
        return result
    
    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


