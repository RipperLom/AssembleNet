
#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: bilm_test.py
Author: wangyan 
Date: 2019/02/13 11:53:24
Brief: 语言模型 测试
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from tfnlp.nets.elmo.bilm import bilm_core
from tfnlp.nets.elmo.bilm import bilm_util
from tfnlp.nets.elmo.data_lm import LMDataSet
from tfnlp.nets.elmo.data_lm import BiLMDataset



def main(args):
    test_prefix = args.test_prefix

    options, ckpt_file = bilm_util.load_options_latest_checkpoint(args.save_dir)
    # load the vocab
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    vocab = bilm_util.load_vocab(args.vocab_file, max_word_length)
    
    kwargs = {
        'test': True,
        'shuffle': False,
    }
    if options.get('bidirectional'):
        data = BiLMDataset(test_prefix, vocab, **kwargs)
    else:
        data = LMDataSet(test_prefix, vocab, **kwargs)

    bilm_core.test(options, ckpt_file, data, batch_size = args.batch_size)
    print("done!")
    return True



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute test perplexity')
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--test_prefix', help='Prefix for test files')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

    args = parser.parse_args()
    main(args)

