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
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import h5py
import numpy as np


g_tfnlp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(g_tfnlp_dir + "/../")

from tfnlp.models.bilm import bilm_model


def dump_emb(embedding_file):
    # Our small dataset.
    raw_context = [
        '石	岩	模	具	架',
        '暖	暖	魔	法	搭	配	赛'
    ]
    tokenized_context = [sentence.split() for sentence in raw_context]
    tokenized_question = [
        ['霄' ,'玉'],
    ]

    # Create the dataset file.
    dataset_file = 'dataset_file.txt'
    with open(dataset_file, 'w') as fout:
        for sentence in tokenized_context + tokenized_question:
            fout.write(' '.join(sentence) + '\n')


    # Location of pretrained LM.  Here we use the test fixtures.
    datadir = os.path.join('model', 'lm_model')
    vocab_file = "data/vocab.txt"
    options_file = os.path.join(datadir, 'options.json')
    weight_file = os.path.join(datadir, 'lm_weights.hdf5')
    vocab_emb_file = "model/lm_model/vocab_embedding.hdf5"


    # Dump the embeddings to a file. Run this once for your dataset.
    bilm_model.dump_bilm_embeddings(
        vocab_file, dataset_file, options_file, weight_file, embedding_file, vocab_emb_file
    )
    return True


out_emb_file = 'elmo_embeddings.hdf5'
#dump_emb(embedding_file)
# Load the embeddings from the file -- here the 2nd sentence.
with h5py.File(out_emb_file, 'r') as fin:
    for key in fin.keys():
        print("key=",key,  " value=", fin[key].shape)
    
    #sents_emb = fin['2'][...]
    #print("emb", sents_emb.shape)

