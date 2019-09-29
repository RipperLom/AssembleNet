#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: bilm_util.py
Author: wangyan 
Date: 2019/02/13 11:53:24
Brief:  bilm 工具
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import time
import json
import h5py
import numpy as np
import tensorflow as tf

from tfnlp.nets.elmo.vocab import Vocabulary
from tfnlp.nets.elmo.vocab import UnicodeCharsVocabulary


def print_variable_summary():
    import pprint
    variables = sorted([[v.name, v.get_shape()] for v in tf.global_variables()])
    pprint.pprint(variables)


# 加载 词<->id 映射词典
def load_vocab(vocab_file, max_word_length=None):
    if max_word_length:
        return UnicodeCharsVocabulary(vocab_file, max_word_length, check = True)
    else:
        return Vocabulary(vocab_file, check = True)


# 加载 配置文件 和 checkpoint
def load_options_latest_checkpoint(tf_save_dir):
    options_file = os.path.join(tf_save_dir, 'options.json')
    ckpt_file = tf.train.latest_checkpoint(tf_save_dir)

    with open(options_file, 'r') as fin:
        options = json.load(fin)
    return options, ckpt_file



