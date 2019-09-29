#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: bilm_core.py
Author: wangyan 
Date: 2019/02/13 11:53:24
Brief: 语言模型 训练
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import logging
import argparse
import numpy as np

from tfnlp.nets.elmo.bilm import bilm_core
from tfnlp.nets.elmo.bilm import bilm_util
from tfnlp.nets.elmo.data_lm import LMDataSet
from tfnlp.nets.elmo.data_lm import BiLMDataset


"""
    options = {
     'bidirectional': False,
      'char_cnn': {
            'activation': 'relu',
            'embedding': {'dim': 16},
            'filters': [[1, 32],
                [2, 32],
                [3, 64],
                [4, 128],
                [5, 256],
                [6, 512],
                [7, 1024]],
        'max_characters_per_token': 10,
        'n_characters': 261,
        'n_highway': 2},
     
     'dropout': 0.1,

     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 8,    #单个tok 向量的维度
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
      
      #轮数
     'n_epochs': 1,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 10,
     #'n_negative_samples_batch': 8192,
     'n_negative_samples_batch': 1,
     
     #add by wangyan
     'word_emb_file':  '/Users/wy/workspace/project/nlp/tfnlp/data/model.vec'
    }
"""


def main(args):
    # load the vocab
    vocab = bilm_util.load_vocab(args.vocab_file, None)

    # define the options
    n_gpus = 1
    batch_size = 2  # batch size for each GPU

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 51
    prefix = args.train_prefix

    options = {
     'bidirectional': True,
     'dropout': 0.1,

     'lstm': {
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'cell_clip': 3,
      'projection_dim': 100,    #单个tok 向量的维度
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
      
      #轮数
     'n_epochs': 1000,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     #'n_negative_samples_batch': 8192,
     'n_negative_samples_batch': 1,
     
     #add by wangyan
     # 'word_emb_file':  '/Users/wy/workspace/project/nlp/tfnlp/data/model.vec'
     # add by GZhao
     'word_emb_file':  '/Users/admin/Desktop/git/oceanus-bot-model'
                       '/data/pretrained/glove100d_add.txt'

    }

    
    if options["bidirectional"]:
        data = BiLMDataset(prefix, vocab, test=False, shuffle=False)
    else:
        data = LMDataSet(prefix, vocab, test=False, shuffle=False)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    bilm_core.train(options, data, n_gpus, tf_save_dir, tf_log_dir)
    return True



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--save_dir', help='Location of checkpoint files')

    args = parser.parse_args()
    main(args)

