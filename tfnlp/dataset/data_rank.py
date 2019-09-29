
#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2018 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: data_rank.py
Author: wangyan(wangyan@aibot.me)
Date: 2018/12/12 11:53:24
Brief: tf 读取
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging
import numpy as np
import tensorflow as tf
import time




def read_pointwise(file_pattern):
    '''
      dataset = dataset.repeat(1).batch(batch_size = 1).prefetch()
      map方法可以接受任意函数以对dataset中的数据进行处理；
      另外，可使用repeat、shuffle、batch、prefetch 方法对dataset进行重复、混洗、分批、预取
      dataset = dataset.repeat(epochs).shuffle(buffer_size).batch(batch_size)
    '''
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=2)
    
    def parser_example(serial_example):
        feature_dic = {}
        feature_dic["label"] = tf.FixedLenFeature([2], tf.int64)
        feature_dic["left"] = tf.FixedLenFeature([32], tf.int64)
        feature_dic["right"] = tf.FixedLenFeature([32], tf.int64)
        feats = tf.parse_single_example(serial_example, features = feature_dic)
        return feats["label"], feats["left"], feats["right"]

    batch_num = 2
    epoches_num = 2
    dataset = dataset.map(map_func=parser_example, num_parallel_calls=2)
    dataset = dataset.repeat(epoches_num).batch(batch_num)

    data_iterator = dataset.make_one_shot_iterator()
    ## make feedable iterator
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
    one_element = data_iterator.get_next()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

   
    with tf.Session() as sess:
        sess.run(init_op)
        train_handle = sess.run(data_iterator.string_handle())

        start_time = time.time()
        try:
            i = 0
            while True:
                i += 1
                #label, left, right = sess.run(one_element)
                label, left, right = sess.run(one_element, feed_dict= {handle: train_handle})
                #print(label, left, right)
                print(i, label)
        except tf.errors.OutOfRangeError:
            print("end!")
    
    duration = time.time() - start_time
    print("duration: %fs, i: %d" % (duration, i))
    return True



### 大数据量
### 优化后的版本
### 1、data并行读取  2、多线程训练
def read_pointwise_good(file_pattern):
    '''
      dataset = dataset.repeat(1).batch(batch_size = 1).prefetch()
      map方法可以接受任意函数以对dataset中的数据进行处理；
      另外，可使用repeat、shuffle、batch、prefetch 方法对dataset进行重复、混洗、分批、预取
      dataset = dataset.repeat(epochs).shuffle(buffer_size).batch(batch_size)
    '''
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=2)
    
    def parser_example(serial_example):
        feature_dic = {}
        feature_dic["label"] = tf.FixedLenFeature([2], tf.int64)
        feature_dic["left"] = tf.FixedLenFeature([32], tf.int64)
        feature_dic["right"] = tf.FixedLenFeature([32], tf.int64)
        feats = tf.parse_single_example(serial_example, features = feature_dic)
        return feats["label"], feats["left"], feats["right"]

    #dataset = dataset.map(map_func=parser_example, num_parallel_calls=4)
    
    '''
    #优化版本
    shuffle_buf = 10000
    epoches_num = 2
    batch_num = 2
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buf, epoches_num))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(parser_example, batch_num))
    #dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/cpu:0"))
    '''
    dataset = dataset.repeat(epoches_num).batch(batch_num)

    data_iterator = dataset.make_one_shot_iterator()
    ## make feedable iterator
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
    one_element = data_iterator.get_next()

    ###多线程训练
    init_op = tf.group(tf.global_variables_initializer(), \
            tf.local_variables_initializer())
    
    thread_num = 4
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=thread_num, 
                                        inter_op_parallelism_threads=thread_num)) \
            as sess:
        
        #初始化
        step = 0
        sess.run(init_op)
        train_handle = sess.run(data_iterator.string_handle())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)

        #开始读取
        start_time = time.time()
        while not coord.should_stop():
            try:
                step += 1
                #label, left, right = sess.run(one_element)
                label, left, right = sess.run(one_element, feed_dict= {handle: train_handle})
                #print(label, left, right)
                print(step, label)

                print("pointwise data read is good")
            except tf.errors.OutOfRangeError:
                print("read %d steps" % step)
                coord.request_stop()
    
    coord.join(threads)
    duration = time.time() - start_time
    print("duration: %fs, step: %d" % (duration, step))
    return True



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



class RankData2TFRecord(object):
    def __init__(self, max_len = 32, pad_id = 0):
        '''
        Args:
            max_len: 最大长度
            pad_id : 填充数字 
        '''
        self.max_len = max_len
        self.pad_id = pad_id
    

    def build_tfrecord(self, file_name, tfrecord_name, task_type):
        if not os.path.isfile(file_name):
            logging.error("the input file not exit!")
            return False
        
        fmt_func = None
        if task_type == "pointwise":
            fmt_func = self.pointwise2example
        elif task_type == "pairwise":
            fmt_func = self.pairwise2example
        else:
            logging.error("task type error! (pointwise or pairwise)")
            return False

        local_writer = tf.python_io.TFRecordWriter(tfrecord_name)
        with open(file_name) as infile:
            for line in infile:
                example = fmt_func(line)
                if example:
                    local_writer.write(example.SerializeToString())
        local_writer.close()
        return True


    def pointwise2example(self, line):
        '''
        Args:
            line: 文本 left_ids \t right_ids \t label(0/1)
        Returns:
            TF example
        '''
        items = line.rstrip("\r\n").split("\t")
        if len(items) != 3:
            logging.warning("the input line fmt error! (left_ids, right_ids, label)")
            return None
        
        label = [0, 0]
        label[int(items[2])] = 1
        feature_dic = {}
        feature_dic["label"] = tf_feature(label, "int64")
        feature_dic["left"] = tf_feature(self._txt2list(items[0]), "int64") 
        feature_dic["right"] = tf_feature(self._txt2list(items[1]), "int64")
        return tf_example(feature_dic) 


    def pairwise2example(self, line):
        '''
        Args:
            line: query_ids \t postitle_ids \t negtitle_ids
        Returns:
            TF example
        '''
        items = line.rstrip("\r\n").split("\t")
        if len(items) != 3:
            logging.warning("the input line fmt error! (query_ids \t pos_ids \t neg_ids)")
            return None

        feature_dic = {}
        feature_dic["left"] = tf_feature(self._txt2list(items[0]), "int64") 
        feature_dic["pos_right"] = tf_feature(self._txt2list(items[1]), "int64")
        feature_dic["neg_right"] = tf_feature(self._txt2list(items[2]), "int64")
        return tf_example(feature_dic) 


    def _txt2list(self, line):
        '''
        text to fix length list
        Args:
            line: txt line
            max_len: the max size of vector
            pad_id: default 0
        '''
        tmp_ids = [int(t) for t in line.strip().split(" ")]
        if len(tmp_ids) < self.max_len:
            add_ids = [self.pad_id] * (self.max_len - len(tmp_ids))
            tmp_ids += add_ids
        return tmp_ids


    def read_pointwise(self, file_list, epochs):
        #file_queue = tf.train.string_input_producer(file_list, num_epochs = epochs)
        pass



def main_test():
    """
    f1 = tf_feature([1,2,3,3,4,5,5,6,6,6,67], "int64")
    f2 = tf_feature([1,2,3,3,4,5,5,6,6,6,67], "float")
    feature_dic = {"label":f1, "data": f2}
    example = tf_example(feature_dic)
    print(example)
    """

    rank2tf = RankData2TFRecord(32, 0)
    rank2tf.build_tfrecord("./data/test_pointwise_data", "./data/test_pointwise_tf", "pointwise")
    rank2tf.build_tfrecord("./data/test_pairwise_data", "./data/test_pairwise_tf", "pairwise")




