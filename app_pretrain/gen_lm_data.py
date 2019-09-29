#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: gen_lm_data.py
Author: wangyan 
Date: 2019/02/13 11:53:24
Brief:  bilm data
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import sys
import time
import json
import logging
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tfnlp.utils.tokenization import BasicTokenizer
from tfnlp.utils.lang_tool  import LangTool
from tfnlp.dataset.data_lm import BiLMDatasetTF


class GenLMData(object):
    def __init__(self):
        self.tok_obj = BasicTokenizer()
        self.seg_obj = LangTool()
        self.dataset_obj = BiLMDatasetTF(max_len=20)


    def load(self, conf_path):
        nlpbase_path = conf_path + "/nlpbase"

        if not self.seg_obj.load(nlpbase_path):
            logging.error("load seg dict error!")
            return False
        
        pretrain_path = conf_path + "/pretrain"
        vocab_file = pretrain_path + "/vocab.txt"
        if not self.dataset_obj.load_vocab(vocab_file):
            logging.error("load vocab dict error!")
            return False
        return True

    
    def process(self, input_file, tok_file, tf_file):
        # clean and seg
        outfile = open(tok_file, "w")
        for line in open(input_file):
            line = line.strip()
            new_line = self.tok_obj.text_norm(line)
            line_seg = self.seg_obj.seg(new_line)
            outfile.write(" ".join(line_seg))
            outfile.write("\n")
        outfile.close()

        # to tf fmt
        return self.dataset_obj.tf_example_file(tok_file, tf_file)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', default='./data/', help='the vocab file')
    parser.add_argument('--input_file', default='./data/pretrain/train.txt', help='train file')
    parser.add_argument('--token_file', default='./data/pretrain/train_tok.txt', help='train token file')
    parser.add_argument('--tf_file', default='./data/pretrain/train_tfrecord', help='train record')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()
    print(args)

    gen_data = GenLMData()
    gen_data.load(args.conf_path)
    gen_data.process(args.input_file, args.token_file, args.tf_file)


