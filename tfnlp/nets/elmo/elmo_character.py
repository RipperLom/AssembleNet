#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2018 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: elmo_character.py
Author: wangyan 
Date: 2019/01/28 11:53:24
Brief: 对整理语料进行 bilm-emb 结果存在文件中
    句子字符输入
    句子tok输入
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import h5py
import numpy as np
import tensorflow as tf

g_tfnlp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(g_tfnlp_dir + "/../")

from tfnlp.models.bilm import bilm_model
from tfnlp.data import Batcher, TokenBatcher
from tfnlp.data import Vocabulary
from tfnlp.models.bilm import bilm_model
from tfnlp.models.bilm import bilm_elmo


# Location of pretrained LM.  Here we use the test fixtures.
datadir = os.path.join('model', 'lm_model')
vocab_file = 'data/vocab.txt'
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'lm_weights.hdf5')

use_char = False
vocab = Vocabulary(vocab_file)
batcher = TokenBatcher(vocab_file)
ids_placeholder = tf.placeholder('int32', shape=(None, None))
emb_file = "model/lm_model/vocab_embedding.hdf5"
model = bilm_model.BiLanguageModel(options_file, weight_file, use_character_inputs=use_char, embedding_weight_file=emb_file)
embeddings_op = model(ids_placeholder)


# Get an op to compute ELMo (weighted average of the internal biLM layers)
# Our SQuAD model includes ELMo at both the input and output layers
# of the task GRU, so we need 4x ELMo representations for the question
# and context at each of the input and output.
# We use the same ELMo weights for both the question and context
# at each of the input and output.
elmo_context_input = bilm_elmo.weight_layers('input', embeddings_op, l2_coef=0.0)
raw_context = [
    '霄	玉',
    '石	岩	模	具	架',
    '暖	暖	魔	法	搭	配	赛',
]
tok_context = [line.split() for line in raw_context]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    for toks in tok_context:
        context_ids = batcher.batch_sents([toks])

        # Compute ELMo representations (here for the input only, for simplicity).
        elmo_context_input_ = sess.run(elmo_context_input['weighted_op'],
            feed_dict={ids_placeholder: context_ids}
        )
        print("xxxx", elmo_context_input_.shape)

