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
Brief: 测试
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import glob
import logging
import numpy as np

g_tfnlp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(g_tfnlp_dir + "/../")

from tfnlp.data import vocab
from tfnlp.data import data_lm


def test_data():
    file_name = "data/vocab.txt"
    vocab_obj = vocab.Vocabulary(file_name, True)
    line = "石  岩  模  具  架"
    sent = vocab_obj.encode(line, False, True)
    print(sent)

    prefix = "data/train.txt"
    #data = data_lm.LMDataSet(prefix, vocab_obj, False, shuffle=False)
    data = data_lm.BiLMDataset(prefix, vocab_obj, False, shuffle=False)

    i = 0
    start = time.time()
    for sent in data.iter_batches(1, 10):
        i += 1
        if i % 80 == 0:
            t_diff = time.time() - start
            print(sent, t_diff, i, "  qps=", i / t_diff)
        if i > 10000:
            break
    return True



def test_tok():
    file_name = "data/vocab.txt"
    data = data_lm.TokenBatcher(file_name)
    sents = [['石', '岩', '模', '具'], ['模', '具', '架']]
    ret = data.batch_sents(sents)
    print(ret)

    data = data_lm.Batcher(file_name, 6)
    ret = data.batch_sents(sents)
    print(ret)



if __name__ == '__main__':
    #test_data()

    test_tok()
    print("done")


