#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 

"""
File: vocab.py
Author: wangyan 
Date: 2019/07/17 11:53:24
Brief:  词典加载
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import glob
import logging
import collections
import numpy as np


PADDING = "<PAD>"
UNKNOWN = "<UNK>"


# 词典 word->id
def build_dict(file_name_list = [], max_size =-1):
    word_counter = collections.Counter()
    for file_name in file_name_list:
        with open(file_name, "r") as infile:
            for line in infile:
                items = line.strip().split()
                word_counter.update(items)

    vocabulary = [key for key,val in word_counter.most_common()]
    vocabulary = [PADDING, UNKNOWN] + vocabulary
    if max_size > 0 and len(vocabulary) > max_size:
        word_counter = vocabulary[0:max_size]
    word2id_dic = dict(zip(vocabulary, range(len(vocabulary))))
    return word2id_dic


# dump word->id 词典
def dump_vocab_dic(word2id_dic, out_file):
    with open(out_file, "w") as outfile:
        for word in word2id_dic:
            outfile.write("%s\t%d\n" % (word, word2id_dic[word]))
    return True


# 加载word->id 词典
def load_vocab_dic(file_name):
    if os.path.isfile(file_name):
        return {}
    word2id_dic = {}
    with open(file_name, "r") as infile:
        for line in infile:
            items = line.rstrip("\r\n").split("\t")
            if len(items) != 2:
                continue
            word2id_dic[items[0]] = int(items[1])
    return word2id_dic


# 样本 word->idx and padded
def example_padded(word2id_dic, sents = [], max_seq_len = 30):
    tok_list = np.zeros((max_seq_len), dtype=np.int32)
    for i in max_seq_len:
        idx = word2id_dic[PADDING]
        if i < len(sents):
            if sents[i] in word2id_dic:
                idx = word2id_dic[sents[i]]
            else:
                idx = word2id_dic[UNKNOWN]
        tok_list[i] = idx
    return tok_list

