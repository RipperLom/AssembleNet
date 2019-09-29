#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: test_dataset_lm.py
Author: wangyan 
Date: 2019/09/25 11:53:24
Brief: 语言模型数据测试
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import glob
import random
import logging

from tfnlp.dataset import data_lm
import unittest


class LMTest(unittest.TestCase):
    #初始化
    @classmethod
    def setUpClass(self):
        self.data_path = "/home/work/boss_bot/oceanus-bot-model/data/pretrain"
        pass
    

    def test_bi_lm(self):
        return 0
        config = {
            "vocab_file": self.data_path + "/vocab.txt",
            "train_prefix": self.data_path + "/train_tok.txt",
            "test": False,
            "shuffle": True
        }
        print(config)

        dataset = data_lm.BiLMDataset(config)
        batches = dataset.ops(2, 6)
        for batch_no, batch in enumerate(batches, start=1):
            print(batch_no)
            print(batch)
            break
        return True
    
    
    def test_bi_lm_tf(self):
        dataset = data_lm.BiLMDatasetTF()
        vocab_file = self.data_path + '/vocab.txt'
        input_file = self.data_path + '/train_tok.txt'
        out_file = self.data_path + '/train.tfrecord'
        
        dataset.load_vocab(vocab_file)
        dataset.tf_example_file(input_file, out_file)


if __name__ == "__main__":
    unittest.main()

