#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: build_w2v.py
Author: wangyan 
Date: 2019/09/05 11:53:24
Brief:  词向量

pip install --upgrade pip
pip install cython
pip install fasttext
pip install gensim
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import gensim
import fasttext


def dump_vocab():
    model = gensim.models.KeyedVectors.load_word2vec_format(fname="./model.vec", binary=False)
    words = model.vocab
    with open("vocab.txt", "w") as f:
        f.write("<S>\n")
        f.write("</S>\n")
        f.write("<UNK>\n")
        for w in words:
            f.write(w)
            f.write("\n")
    return True



def train_w2v(train_file, model_file):
    # doc: https://pypi.org/project/fasttext/
    model = fasttext.skipgram(train_file, model_file, \
            lr=0.01, dim=256, \
            min_count=1, thread=30, \
            t=1e-4, ws=5, neg=5, \
            epoch=10, silent=False)
    #        min_count=5, thread=30, t=1e-4, ws=5, neg=5, epoch=10, silent=False)


if __name__ == "__main__":
    train_file = "train.txt"
    model_file = "model"
    train_w2v(train_file, model_file)

