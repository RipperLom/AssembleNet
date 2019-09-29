#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: data_lm.py
Author: wangyan 
Date: 2019/04/11 11:53:24
Brief: 语言模型数据读写
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import glob
import random
import logging
import argparse
import collections

import json
import numpy as np
import tensorflow as tf

from tfnlp.nets.elmo.bilm import bilm_util
from tfnlp.nets.elmo.data_lm import LMDataSet


class BiLMDataset(object):
    '''
    双向语言模型数据
    '''

    def __init__(self, config):
        '''
        bidirectional version of LMDataset
        '''
        self.config = config
        self.vocab = bilm_util.load_vocab(self.config['vocab_file'], None)
        self.file_pat = config['train_prefix']
        self.test = config['test']
        self.shuffle = config['shuffle']
        self._data_forward = LMDataSet(self.file_pat, self.vocab,
                                       False, self.test, self.shuffle)
        self._data_reverse = LMDataSet(self.file_pat, self.vocab,
                                       True, self.test, self.shuffle)

    def ops(self, batch_size, num_steps):
        '''
            双向语言模型迭代输入 生成数据
        '''
        for X, Xr in \
            zip(self._data_forward.iter_batches(batch_size, num_steps),
                self._data_reverse.iter_batches(batch_size, num_steps)):
            for k, v in Xr.items():
                X[k + '_reverse'] = v
            yield X


class BiLMDatasetTF(object):
    """
        bidirectional language model 数据处理
    """
    def __init__(self, max_len=20):
        self.pad_id = 0
        self.unk_id = 1
        self.begin_id = 2
        self.end_id = 3

        self.max_len = max_len
        self.id2vocab = {}
        self.vocab2id = {}


    def load_vocab(self, vocab_file):
        '''
        load vocab_file
        Args:
            vocab_file: path of vocab
        Returns:
            True / False
        Raises:
            None
        '''
        with open(vocab_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                items = line.strip().split("\t")
                if len(items) != 2:
                    continue
                key = items[0]
                val = int(items[1])
                self.vocab2id[key] = val
                self.id2vocab[val] = key
        return True

    def convert_tokens2sample(self, tok_list):
        '''
        covert tok_list to tf sample object
        Args:
            tok_list: sentence cut by words
        Returns:
            tf sample
        Raises:
            None
        '''
        if len(self.vocab2id) == 0:
            logging.error("check vocab dict is empty")
            return None
        
        ids, next_ids = self.sample2id_pad(tok_list, reverse=False)
        ids_reverse, next_ids_reverse = self.sample2id_pad(tok_list, reverse=True)

        # print(ids, next_ids)
        # print(ids_reverse, next_ids_reverse)
        example = self.tf_example(ids, next_ids, ids_reverse, next_ids_reverse)
        return example

    def sample2id_pad(self, x_list=[], reverse = False):
        '''
        covert words to ids
        Args:
            x_list: sentence cut by words
            reverse: reverse sents
        Returns:
            token_ids / next_token_ids
        Raises:
            None
        '''
        # gen x_ids: length (max_len + 1)
        if reverse:
            x_list.reverse()

        x_ids = [0] * (self.max_len + 1)
        x_ids[0] = self.begin_id

        i = 1
        for w in x_list:
            x_ids[i] = self.vocab2id.get(w, self.unk_id)
            i += 1
            # 长度已经达到 1 + max_len
            if i > self.max_len:
                break
        else:  # i <= self.max_len + 1
            x_ids[i] = self.end_id

        token_ids = x_ids[: -1]
        next_token_ids = x_ids[1:]
        return token_ids, next_token_ids

    def tf_example(self, token_ids, next_token_ids,
                   token_ids_reverse, next_token_ids_reverse):
        '''covert int_list to tf.data
        Args:
            token_ids: word ids
            token_ids_reverse: word ids
            next_token_ids: next word ids
            next_token_ids_reverse: next word ids
        Returns:
            tf.train.Example
        Raises:
            None
        '''
        def feat_int64(tok_list):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=tok_list))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={"token_ids": feat_int64(token_ids),
                    "token_ids_reverse": feat_int64(token_ids_reverse), 
                    "next_token_ids_reverse": feat_int64(next_token_ids_reverse),
                    "next_token_ids": feat_int64(next_token_ids) }))
        return example

    def tf_example_file(self, input_file, out_file):
        """
        tf data 格式转换
        Args:
            input_prefix: 输入文件
            out_prefix  : 输出前缀
        Returns:
            None
        """
        fwriter = tf.python_io.TFRecordWriter(out_file)
        num = 0
        with open(input_file, "r") as infile:
            for line in infile:
                items = line.strip().split()
                if len(items) == 0:
                    continue
                
                example = self.convert_tokens2sample(items)
                if example:
                    fwriter.write(example.SerializeToString())
                else:
                    print("error_line:\t" + line.strip())
                
                num += 1
                if (num % 10000 == 0):
                    print("example proc %d ..." % (num))
        fwriter.close()
        return True

