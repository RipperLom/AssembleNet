#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2018 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: test_data.py
Author: wangyan 
Date: 2019/01/28 11:53:24
Brief: 对整理语料进行 bilm-emb 结果存在文件中

ELMo usage example with pre-computed and cached context independent
token representations
Below, we show usage for SQuAD where each input example consists of both
a question and a paragraph of context.

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

from tfnlp.data import TokenBatcher
from tfnlp.models.bilm import bilm_model 
from tfnlp.models.bilm import bilm_elmo
from tfnlp.models.bilm import bilm_model


# Our small dataset.
raw_context = [
    '石	岩	模	具	架',
    '暖	暖	魔	法	搭	配	赛'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['霄' ,'玉'],
]

# Create the vocabulary file with all unique tokens and
# the special <S>, </S> tokens (case sensitive).
all_tokens = set(['<S>', '</S>'] + tokenized_question[0])
for context_sentence in tokenized_context:
    for token in context_sentence:
        all_tokens.add(token)
vocab_file = 'vocab_small.txt'
with open(vocab_file, 'w') as fout:
    fout.write('\n'.join(all_tokens))


# Location of pretrained LM.  Here we use the test fixtures.
datadir = os.path.join('model', 'lm_model')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'lm_weights.hdf5')


# Dump the token embeddings to a file. Run this once for your dataset.
token_embedding_file = 'elmo_token_embeddings.hdf5'
bilm_model.dump_token_embeddings(
    vocab_file, options_file, weight_file, token_embedding_file
)
tf.reset_default_graph()


## Now we can do inference.
# Create a TokenBatcher to map text to token ids.
batcher = TokenBatcher(vocab_file)

# Input placeholders to the biLM.
context_token_ids = tf.placeholder('int32', shape=(None, None))
question_token_ids = tf.placeholder('int32', shape=(None, None))

# Build the biLM graph.
bilm = bilm_model.BiLanguageModel(
    options_file,
    weight_file,
    use_character_inputs=False,
    embedding_weight_file=token_embedding_file
)

# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_token_ids)
question_embeddings_op = bilm(question_token_ids)

# Get an op to compute ELMo (weighted average of the internal biLM layers)
# Our SQuAD model includes ELMo at both the input and output layers
# of the task GRU, so we need 4x ELMo representations for the question
# and context at each of the input and output.
# We use the same ELMo weights for both the question and context
# at each of the input and output.
elmo_context_input = bilm_elmo.weight_layers('input', context_embeddings_op, l2_coef=0.0)
with tf.variable_scope('', reuse=True):
    # the reuse=True scope reuses weights from the context for the question
    elmo_question_input = bilm_elmo.weight_layers(
        'input', question_embeddings_op, l2_coef=0.0
    )

elmo_context_output = bilm_elmo.weight_layers(
    'output', context_embeddings_op, l2_coef=0.0
)
with tf.variable_scope('', reuse=True):
    # the reuse=True scope reuses weights from the context for the question
    elmo_question_output = bilm_elmo.weight_layers(
        'output', question_embeddings_op, l2_coef=0.0
    )


with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    context_ids = batcher.batch_sentences(tokenized_context)
    question_ids = batcher.batch_sentences(tokenized_question)

    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_context_input_, elmo_question_input_ = sess.run(
        [elmo_context_input['weighted_op'], elmo_question_input['weighted_op']],
        feed_dict={context_token_ids: context_ids,
                   question_token_ids: question_ids}
    )


