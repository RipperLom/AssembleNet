#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: dataset_cls.py
Author: wangyan 
Date: 2019/09/23 11:53:24
Brief: 分类任务数据 读写
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import json
import logging
import argparse
import collections

import numpy as np
import tensorflow as tf


class DateSetCLS(object):
    """
        分类数据处理
    """
    def __init__(self, max_len = 32):
        self.pad_id = 0
        self.unk_id = 1
        self.begin_id = 2
        self.end_id = 3

        self.n_label = 0
        self.max_len = max_len
        self.label2id = {}
        self.id2label = []

        self.id2vocab = {}
        self.vocab2id = {}
        pass
    

    def gen_vocab(self, data_file, vocab_file, threshold = 0):
        '''
        Args:
            data_file: 主题对应id的字典
        Returns:
            None
        Raises:
            None
        '''
        with open(data_file, "r") as infile:
            kv_dic = collections.Counter()
            for line in infile:
                items = line.strip().split("\t")
                for w in items[1:]:
                    w = w.strip()
                    if len(w) == 0:
                        continue
                    kv_dic[w] += 1
        
        # dump vocab
        outfile = open(vocab_file, "w")
        outfile.write("PAD\t0\n")
        outfile.write("UNK\t1\n")
        outfile.write("<S>\t2\n")
        outfile.write("</S>\t3\n")
        
        ID = 10
        for k,v in kv_dic.most_common():
            if v < threshold:
                continue
            outfile.write("%s\t%d\n" % (k, ID))
            ID += 1
        outfile.close()
        return True
    

    def load(self, label_file, vocab_file):
        '''
        Args:
            label_file: 标准类别 
            vocab_file: 词典
        Returns:
            True / False
        Raises:
            None
        '''
        # load label
        with open(label_file, 'r') as infile:
            # 格式：label_1 \t label_2 \t id
            ID = 0
            for line in infile:
                items = line.strip().split('\t')
                if len(items) != 3:
                    continue
                label_1 = items[0]
                label_2 = items[1]
                label_id = items[2]
                self.id2label.append((label_1, label_2, label_id))
                self.label2id[label_id] = ID
                ID += 1
            self.n_label = len(self.label2id)
        
        # load vocab
        with open(vocab_file, "r") as infile:
            # 格式：word \t id
            for line in infile:
                items = line.strip().split("\t")
                if len(items) != 2:
                    continue
                ID = int(items[1])
                self.vocab2id[items[0]] = ID
                self.id2vocab[ID] = items[0]
        return True


    def sample2id_pad(self, y, x_list = []):
        if y not in self.label2id:
            logging.error("label=%s error!", y)
            return [], []
        
        # gen y
        y_id = self.label2id[y]
        y_ids = [0] * self.n_label
        y_ids[y_id] = 1

        # gen x
        if len(x_list) > self.max_len - 2:
            x_list = x_list[0: self.max_len - 2]
        x_ids = [0] * self.max_len
        x_ids[0] = self.begin_id
        x_ids[-1] = self.end_id
        i = 1
        for w in x_list:
            x_ids[i] = self.vocab2id.get(w, self.unk_id)
            i += 1
        return y_ids, x_ids
    

    def tf_example(self, y_ids, x_ids):
        example = tf.train.Example(
                features=tf.train.Features(
                feature={"label": tf.train.Feature(int64_list=tf.train.Int64List(value=y_ids)), \
                        "left": tf.train.Feature(int64_list=tf.train.Int64List(value=x_ids)) }))
        return example
    

    def tf_example_file(self, input_file, out_file):
        """
        tf data 格式转换
        Args:
            input_prefix: 输入文件
            out_prefix  : 输出前缀
            out_num     : 输出文件个数
        Returns:
            None
        """
        fwriter = tf.python_io.TFRecordWriter(out_file)

        num = 0
        for line in open(input_file, "r"):
            items = line.strip().split("\t", 1)
            if len(items) != 2:
                continue
            num += 1
            # label \t w1  w2  w3
            y = items[0]
            X = items[1].split("\t")
            y_ids, x_ids = self.sample2id_pad(y, X)
            if len(y_ids) == 0 or len(x_ids) == 0:
                logging.error("convert word to id error! line=" + line)
                continue
            
            example = self.tf_example(y_ids, x_ids)
            fwriter.write(example.SerializeToString())
            if (num % 10000 == 0):
                print("example proc %d ..." %(num))
        fwriter.close()
        return True




