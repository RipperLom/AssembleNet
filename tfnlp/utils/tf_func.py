#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: tf_func.py
Author: wangyan 
Date: 2019/09/10 11:53:24
Brief: tf 相关工具
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import pprint

import tensorflow as tf


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

def print_variable():
    """
        print variable name and shape
    """
    variables = sorted([[v.name, v.get_shape()] for v in tf.global_variables()])
    pprint.pprint(variables)
    return True

def load_latest_checkpoint(tf_save_dir):
    """
    load checkpoint
    """
    ckpt_file = tf.train.latest_checkpoint(tf_save_dir)
    return ckpt_file


def seq_length(sequence):
    """
    get sequence length
    for id-sequence, (N, S)
        or vector-sequence  (N, S, D)
    """
    if len(sequence.get_shape().as_list()) == 2:
        used = tf.sign(tf.abs(sequence))
    else:
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def get_cross_mask(seq1, seq2):
    """
    get matching matrix mask, for two sequences( id-sequences or vector-sequences)
    """
    length1 = seq_length(seq1)
    length2 = seq_length(seq2)
    max_len1 = tf.shape(seq1)[1]
    max_len2 = tf.shape(seq2)[1]
    ##for padding left
    mask1 = tf.sequence_mask(length1, max_len1, dtype=tf.int32)
    mask2 = tf.sequence_mask(length2, max_len2, dtype=tf.int32)
    cross_mask = tf.einsum('ij,ik->ijk', mask1, mask2)
    return cross_mask

