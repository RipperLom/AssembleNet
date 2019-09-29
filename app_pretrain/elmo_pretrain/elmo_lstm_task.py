#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/9/23 8:13 PM
# software: PyCharm


import os
import sys
import copy
import time
import random

import json
import h5py
import numpy as np
import tensorflow as tf

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir + "/../..")

from tfnlp.utils.common import clazz
from tfnlp.utils.common import load_config

class LstmElmoTask(object):
    '''
        ELMO-LSTM网络 预训练任务
    '''
    def __init__(self):
        self.config = {}
        self.trainer = None
        pass


    def load(self, conf_path):
        conf_file = conf_path
        self.config = load_config(conf_file)
        self.dev_epochs = int(int(self.config["data_size"]) * self.config['num_epochs']
            / int(self.config["batch_size"])) * self.config['dev_data_size']
                              #训练集的最大批次数（尽可能大）
        self.options = {
            'bidirectional': self.config['bidirectional'],
            'dropout': self.config['dropout'],
            'lstm': {
                'dim': self.config['dim'],
                'n_layers': self.config['n_layers'],
                'proj_clip': self.config['proj_clip'],
                'cell_clip': self.config['cell_clip'],
                'projection_dim': self.config['projection_dim'],  # 单个tok 向量的维度
                'use_skip_connections': self.config['use_skip_connections']},
            'all_clip_norm_val': self.config['all_clip_norm_val'],
            # 轮数
            'n_epochs': self.config['num_epochs'],
            # number of tokens in training data (this for 1B Word Benchmark)
            'n_train_tokens': self.config['n_train_tokens'],
            'batch_size': self.config['batch_size'],
            'n_tokens_vocab': self.config['n_tokens_vocab'],
            'unroll_steps': self.config['unroll_steps'],
            'n_negative_samples_batch': self.config['n_negative_samples_batch'],
            'word_emb_file': self.config['word_emb_file']
        }
        with open(os.path.join(self.config['save_json_dir']), 'w') as fout:
            fout.write(json.dumps(self.options))

        # 选择类进行组装
        # dataset = clazz(self.config, 'dataset_py', 'dataset_class')(self.config)
        self.transform = clazz(self.config, 'transform_py', 'transform_class')(self.config)
        self.net = clazz(self.config, 'net_py', 'net_class')(self.config)
        self.loss = clazz(self.config, "loss_py", "loss_class")(self.config)
        self.optimizer = clazz(self.config, 'optimizer_py', 'optimizer_class')(self.config)
        self.trainer = clazz(self.config, "trainer_py", "trainer_class")(self.config,
                        self.transform, self.net, self.loss, self.optimizer)


    def predict(self, query, query_seg = []):
        return


    def train(self):
        # 获得目标损失
        token_ids, token_ids_reverse, next_token_ids, next_token_ids_reverse = self.transform.ops()
        lstm_outputs = self.net.ops(token_ids['token_ids'], token_ids_reverse['token_ids_reverse'])
        next_ids = [next_token_ids['next_token_ids'], next_token_ids_reverse['next_token_ids_reverse']]

        train_loss = self.loss.ops(lstm_outputs, next_ids)

        dev_config = copy.deepcopy(self.config)
        dev_config.update({"num_epochs": self.dev_epochs,
                           "train_file": dev_config["dev_file"]})
        dev_transform = clazz(dev_config, 'transform_py', 'transform_class')(dev_config)
        dev_ids, dev_ids_reverse, next_dev_ids, next_dev_ids_reverse = dev_transform.ops()
        dev_lstm_outputs = self.net.ops(dev_ids['token_ids'], dev_ids_reverse['token_ids_reverse'])
        dev_next_ids = [next_dev_ids['next_token_ids'], next_dev_ids_reverse['next_token_ids_reverse']]
        dev_loss = self.loss.ops(dev_lstm_outputs, dev_next_ids)

        # train
        self.trainer.train(self.config, train_loss, -1 * dev_loss)
        return True


    def test(self):
        self.trainer.predict(self.config)

if __name__ == "__main__":
    print("done")

