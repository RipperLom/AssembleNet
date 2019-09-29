#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: tf_record.py
Author: wangyan(wangyan@aibot.me)
Date: 2019/09/04 11:53:24
Brief: tf 数据转换
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging
import random

import numpy as np
import tensorflow as tf


def tf_feature(values, data_type):
    '''
    create new feature by list or tuple data

    Args:
        values: A scalar or list of values
        data_type: int64、bytes、float
    Returns:
        A TF-Feature
    '''
    if not isinstance(values, (tuple, list)):
        values = [values]
    
    if data_type == "int64":
        return tf.train.Feature(int64_list = tf.train.Int64List(value = values))
    elif data_type == "float":
        return tf.train.Feature(float_list = tf.train.FloatList(value = values))
    elif data_type == "bytes":
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = values))
    else:
        logging.error("tf_feature data type error!")
        return None


def tf_example(feature_dic):
    '''
    tf example pb: 协议
        message Example {
            Features features = 1;
        };

        message Features {
           map<string, Feature> feature = 1;
        };

        message Feature {
            oneof kind {
                BytesList bytes_list = 1;
                FloatList float_list = 2;
                Int64List int64_list = 3;
            }
        };
    
    Args:
        feature_dic: key/val
            key: string
            val: TF-Feature
    Returns:
        TF-example
    '''
    return tf.train.Example(features = tf.train.Features(feature=feature_dic))




## just for test
def gen_sample(output_dir):
    i = 0
    for i in range(10):
        name = "%s/part_%05d" % (output_dir, i)
        print("write to file " + name)

        local_writer = tf.python_io.TFRecordWriter(name)
        for j in range(10):
            feature_dic = {}
            data = list(np.random.rand(100))
            feature_dic["data"] = tf_feature(data, "float")
            feature_dic["label"] = tf_feature([random.randint(1,10)], "int64")
            example = tf_example(feature_dic)
            local_writer.write(example.SerializeToString())
        local_writer.close()
    return True

