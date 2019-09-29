#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/9/27 9:20 PM
# software: PyCharm


import sys
import time
import copy

import tensorflow as tf

from tfnlp.utils.common import clazz


class ClassicTrainer(object):

    def __init__(self, config, transform, net, loss, optimizer):
        self.thread_num = int(config["thread_num"])
        self.model_path = config["model_path"]
        self.model_file = config["model_prefix"]
        self.print_iter = int(config["print_iter"])
        self.data_size = int(config["data_size"])
        self.batch_size = int(config["batch_size"])
        self.epoch_iter = int(self.data_size / self.batch_size)
        self.mode = config["training_mode"]

        self.optimizer = optimizer
        self.transform = transform
        self.net = net
        self.loss = loss

    def train(self, config, loss):
        """
        train
        """
        # define optimizer
        optimizer = self.optimizer.ops().minimize(loss)

        saver = tf.train.Saver(max_to_keep=None)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        avg_cost = 0.0
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.thread_num,
                                              inter_op_parallelism_threads=self.thread_num)) as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            read_thread = tf.train.start_queue_runners(sess=sess, coord=coord)
            step = 0
            epoch_num = 1
            start_time = time.time()
            while not coord.should_stop():
                try:
                    step += 1
                    c, _ = sess.run([loss, optimizer])
                    avg_cost += c

                    if step % self.print_iter == 0:
                        print("loss: %f" % ((avg_cost / self.print_iter)))
                        avg_cost = 0.0
                    if step % self.epoch_iter == 0:
                        end_time = time.time()
                        print("save model epoch%d, used time: %d" % (epoch_num, end_time - start_time))
                        save_path = saver.save(sess, "%s/%s.epoch%d" % (self.model_path, self.model_file, epoch_num))
                        epoch_num += 1
                        start_time = time.time()

                except tf.errors.OutOfRangeError:
                    save_path = saver.save(sess, "%s/%s.final" % (self.model_path, self.model_file))
                    coord.request_stop()
            coord.join(read_thread)
        sess.close()

    def predict(self, config, label_index, pred):
        """
        predict
        """
