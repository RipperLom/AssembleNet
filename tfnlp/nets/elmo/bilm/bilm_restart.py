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

from tfnlp.models.bilm import bilm_core
from tfnlp.models.bilm import bilm_util
from tfnlp.data import LMDataSet
from tfnlp.data import BiLMDataset


def main(args):
    options, ckpt_file = bilm_util.load_options_latest_checkpoint(args.save_dir)

    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    vocab = bilm_util.load_vocab(args.vocab_file, max_word_length)

    prefix = args.train_prefix

    kwargs = {
        'test': False,
        'shuffle': True,
    }

    if options.get('bidirectional'):
        data = BiLMDataset(prefix, vocab, **kwargs)
    else:
        data = LMDataset(prefix, vocab, **kwargs)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir

    # set optional inputs
    if args.n_train_tokens > 0:
        options['n_train_tokens'] = args.n_train_tokens
    if args.n_epochs > 0:
        options['n_epochs'] = args.n_epochs
    if args.batch_size > 0:
        options['batch_size'] = args.batch_size

    bilm_core.train(options, data, args.n_gpus, tf_save_dir, tf_log_dir,
          restart_ckpt_file=ckpt_file)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--n_gpus', type=int, default=1,  help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--n_train_tokens', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=0)

    args = parser.parse_args()
    main(args)

