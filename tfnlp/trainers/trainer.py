#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/8/29 8:25 PM
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

        config.update({"num_epochs": "1", "batch_size": "1", "shuffle": "0", "train_file": config["test_file"]})
        transform = clazz(config, 'transform_py', 'transform_class')(config)

        if self.mode == "pointwise":
            test_l, test_r, label = transform.ops()
            # test network
            pred = self.net.predict(test_l, test_r)
        elif self.mode == "pairwise":
            test_l, test_r, label = transform.ops()
            # test network
            pred = self.net.predict(test_l, test_r)
        elif self.mode == 'listwise':
            input_l, label = transform.ops()
            pred = self.net.predict(input_l, input_l)
        else:
            print(sys.stderr, "training mode not supported")
            sys.exit(1)

        mean_acc = 0.0
        saver = tf.train.Saver()
        label_index = tf.argmax(label, 1)
        if self.mode == "pointwise":
            pred_prob = tf.nn.softmax(pred, -1)
            score = tf.reduce_max(pred_prob, -1)
            pred_index = tf.argmax(pred_prob, 1)
            correct_pred = tf.equal(pred_index, label_index)
            acc = tf.reduce_mean(tf.cast(correct_pred, "float"))
        elif self.mode == "pairwise":
            score = pred
            pred_index = tf.argmax(pred, 1)
            acc = tf.constant([0.0])
        elif self.mode == 'listwise':
            pred_prob = tf.nn.softmax(pred, -1)
            score = tf.reduce_max(pred_prob, -1)
            pred_index = tf.argmax(pred_prob, 1)
            correct_pred = tf.equal(pred_index, label_index)
            acc = tf.reduce_mean(tf.cast(correct_pred, "float"))

        correct_pred = tf.equal(pred_index, label_index)
        acc = tf.reduce_mean(tf.cast(correct_pred, "float"))

        modelfile = config["test_model_file"]
        result_file = open(config["test_result"], "w")
        step = 0
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1)) \
                as sess:
            sess.run(init)
            saver.restore(sess, modelfile)
            coord = tf.train.Coordinator()
            read_thread = tf.train.start_queue_runners(sess=sess, coord=coord)
            while not coord.should_stop():
                step += 1
                try:
                    ground, pi, a, prob = sess.run([label_index, pred_index, acc, score])
                    mean_acc += a
                    for i in range(len(prob)):
                        result_file.write("%s\t%d\t%d\t%f\t%s\n" %
                                          ('query', ground[i], pi[i], prob[i], 'info'))
                except tf.errors.OutOfRangeError:
                    coord.request_stop()
            coord.join(read_thread)
        sess.close()
        result_file.close()
        # if self.mode in ["pointwise", 'listwise']:
        #     mean_acc = mean_acc / step
        #     print(sys.stderr, "accuracy: %4.2f" % (mean_acc * 100))


class StandardTrainer(object):
    '''
    pick model with bast performance in dev data
    and predict in test data
    '''

    def __init__(self, config, transform, net, loss, optimizer):
        '''
        n op to compute diff between two sentence representations.
        Args:
            config：配置文件中的超参数
            transform：从转换的tf.data文件，读取batch
            net：由layer和block拼装成的网络，得到预测值
            loss：构建代价函数（Listwise、Pointwise、Pairwise）
            optimazer：优化器
        Returns:
            None
        Raises:
            None
        '''
        self.early_stop_threshold = config["early_stop_threshold"] #提前终止的阈值
        self.warm_up = int(self.early_stop_threshold / 10) #预热，不保存模型
        self.best_acc = 0 #初始化最好的准确率
        self.best_step = 0 #初始化最优准确率的批次
        self.thread_num = int(config["thread_num"]) #线程数量
        self.model_path = config["model_path"] #保存模型的路径
        self.model_file = config["model_prefix"] #保存模型的名字
        self.print_iter = int(config["print_iter"]) #定步长观察训练
        self.data_size = int(config["data_size"]) #训练集数据量
        self.batch_size = int(config["batch_size"]) #批次大小
        self.epoch_iter = int(self.data_size / self.batch_size)  #训练集一个epoch有多少batch
        self.mode = config["training_mode"] #训练模式
        self.dev_data_size = config['dev_data_size'] #验证集数据量
        self.batches_per_epoch = self.dev_data_size // self.batch_size #验证集一个epoch有多少batch
        self.dev_epochs = int(self.data_size * config['num_epochs'] #训练集的最大批次数（尽可能大）
                              / self.batch_size) * self.dev_data_size

        self.optimizer = optimizer
        self.transform = transform
        self.net = net
        self.loss = loss

    def train(self, config, loss, acc):
        """
        train
        """
        # # train
        # if self.mode == "pointwise":
        #     train_l, train_r, train_label = self.transform.ops()
        #     train_pred = self.net.predict(train_l, train_r)
        #     loss = self.loss.ops(train_pred, train_label)
        # elif self.mode == "pairwise":
        #     train_l, pos_train, neg_train = self.transform.ops()
        #     pos_score = self.net.predict(train_l, pos_train)
        #     neg_score = self.net.predict(train_l, neg_train)
        #     loss = self.loss.ops(pos_score, neg_score)
        # elif self.mode == 'listwise':
        #     train_l, train_label = self.transform.ops()
        #     train_pred = self.net.predict(train_l, train_l)
        #     loss = self.loss.ops(train_pred, train_label)
        # else:
        #     print(sys.stderr, "training mode not supported")
        #     sys.exit(1)
        #
        # # dev
        # dev_config = copy.deepcopy(config)
        # dev_config.update({"num_epochs": self.dev_epochs,
        #                    "train_file": dev_config["dev_file"]})
        # dev_transform = clazz(dev_config, 'transform_py', 'transform_class')(dev_config)
        #
        # if self.mode in ["pointwise", 'pairwise']:
        #     dev_l, dev_r, dev_label = dev_transform.ops()
        #     dev_pred = self.net.predict(dev_l, dev_r)
        # elif self.mode == 'listwise':
        #     dev_l, dev_label = dev_transform.ops()
        #     dev_pred = self.net.predict(dev_l, dev_l)
        # else:
        #     print(sys.stderr, "training mode not supported")
        #     sys.exit(1)
        #
        # label_index = tf.argmax(dev_label, 1)
        # if self.mode in ["pointwise", 'listwise']:
        #     pred_prob = tf.nn.softmax(dev_pred, -1)
        #     score = tf.reduce_max(pred_prob, -1)
        #     pred_index = tf.argmax(pred_prob, 1)
        #     correct_pred = tf.equal(pred_index, label_index)
        #     acc = tf.reduce_mean(tf.cast(correct_pred, "float"))
        # elif self.mode == "pairwise":
        #     score = dev_pred
        #     pred_index = tf.argmax(dev_pred, 1)
        #     acc = tf.constant([0.0])
        # else:
        #     print(sys.stderr, "training mode not supported")
        #     sys.exit(1)

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

                    # 观察状态、保存模型
                    if step % self.print_iter == 0:
                        avg_acc = 0

                        for i in range(self.batches_per_epoch):
                            avg_acc += sess.run(acc)
                        avg_acc /= self.batches_per_epoch
                        print("Epoch:%d\t Step:%i\t train loss:%f\t dev acc or loss:%f"
                              % (epoch_num, step, avg_cost / self.print_iter, avg_acc), end='\t')

                        # 选择最优模型
                        upgrade = avg_acc - self.best_acc
                        if upgrade > 0:
                            self.best_acc = avg_acc
                            self.best_step = step

                            if step > self.warm_up:
                                save_path = saver.save(sess,
                                "%s/%s.best" % (self.model_path, self.model_file))
                                print("*\tSaving model")
                            else:
                                print('*')
                        else:
                            print()
                        avg_cost = 0.0

                    # 记录批次
                    if step % self.epoch_iter == 0:
                        end_time = time.time()
                        print("epoch%d, used time: %d" % (epoch_num, end_time - start_time))
                        # save_path = saver.save(sess, "%s/%s.epoch%d" % (self.model_path, self.model_file, epoch_num))
                        epoch_num += 1
                        start_time = time.time()

                    # 过拟合，提前截止
                    if step > self.best_step + self.early_stop_threshold:
                        print("Best dev: %s" % (self.best_acc))
                        print("early stopping")
                        break

                except tf.errors.OutOfRangeError:
                    save_path = saver.save(sess, "%s/%s.final" % (self.model_path, self.model_file))
                    coord.request_stop()
            coord.join(read_thread)
        sess.close()

    def predict(self, config):
        """
        predict
        """

        config.update({"num_epochs": "1", "batch_size": "1", "shuffle": "0", "train_file": config["test_file"]})
        transform = clazz(config, 'transform_py', 'transform_class')(config)

        if self.mode == "pointwise":
            test_l, test_r, label = transform.ops()
            # test network
            pred = self.net.predict(test_l, test_r)
        elif self.mode == "pairwise":
            test_l, test_r, label = transform.ops()
            # test network
            pred = self.net.predict(test_l, test_r)
        elif self.mode == 'listwise':
            input_l, label = transform.ops()
            pred = self.net.predict(input_l, input_l)
        else:
            print(sys.stderr, "training mode not supported")
            sys.exit(1)

        mean_acc = 0.0
        saver = tf.train.Saver()
        label_index = tf.argmax(label, 1)
        if self.mode == "pointwise":
            pred_prob = tf.nn.softmax(pred, -1)
            score = tf.reduce_max(pred_prob, -1)
            pred_index = tf.argmax(pred_prob, 1)
            correct_pred = tf.equal(pred_index, label_index)
            acc = tf.reduce_mean(tf.cast(correct_pred, "float"))
        elif self.mode == "pairwise":
            score = pred
            pred_index = tf.argmax(pred, 1)
            acc = tf.constant([0.0])
        elif self.mode == 'listwise':
            pred_prob = tf.nn.softmax(pred, -1)
            score = tf.reduce_max(pred_prob, -1)
            pred_index = tf.argmax(pred_prob, 1)
            correct_pred = tf.equal(pred_index, label_index)
            acc = tf.reduce_mean(tf.cast(correct_pred, "float"))

        modelfile = config["test_model_file"]
        result_file = open(config["test_result"], "w")
        step = 0
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1)) \
                as sess:
            sess.run(init)
            saver.restore(sess, modelfile)
            coord = tf.train.Coordinator()
            read_thread = tf.train.start_queue_runners(sess=sess, coord=coord)
            while not coord.should_stop():
                step += 1
                try:
                    ground, pi, a, prob = sess.run([label_index, pred_index, acc, score])
                    mean_acc += a
                    for i in range(len(prob)):
                        result_file.write("%s\t%d\t%d\t%f\t%s\n" %
                                          ('query', ground[i], pi[i], prob[i], 'info'))
                except tf.errors.OutOfRangeError:
                    coord.request_stop()
            coord.join(read_thread)
        sess.close()
        result_file.close()
        if self.mode in ["pointwise", 'listwise']:
            mean_acc = mean_acc / step
            print(sys.stderr, "accuracy: %4.2f" % (mean_acc * 100))


