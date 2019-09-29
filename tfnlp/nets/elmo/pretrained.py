#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/9/23 7:57 PM
# software: PyCharm


import logging

import gensim
import numpy as np
import tensorflow as tf

from tfnlp import layers
from tfnlp import blocks


class ElmoLstm(object):

    '''
    构建tf计算图 for NLMs
    所有的超参都在config中
    '''

    def __init__(self, config):
        self.DTYPE = 'float32'
        self.DTYPE_INT = 'int64'
        self.config = config
        self.n_tokens_vocab = self.config['n_tokens_vocab']
        self.projection_dim = self.config['projection_dim']

        self.save_json_dir = self.config['save_json_dir']
        self.block_lstm = blocks.BlockLSTM(self.save_json_dir)


    def ops(self, token_ids, token_ids_reverse):
        # 通过word2vec 进行初始化embedding_weights
        emb_file = self.config.get("word_emb_file", "")
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(fname=emb_file)
        tmp_embd = np.zeros((self.n_tokens_vocab, self.projection_dim), dtype=self.DTYPE)
        words = w2v_model.vocab

        for i, word in enumerate(words):
            tmp_embd[i + 4][:] = w2v_model[word]
        print("load word2vec file for word embedding")

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            embedding_weights = tf.get_variable(
                "embedding", [self.n_tokens_vocab, self.projection_dim],
                dtype=self.DTYPE,
                initializer=tf.constant_initializer(tmp_embd),
                trainable=True
            )

            # self._build(token_ids, token_ids_reverse)
            output_tensor = self.block_lstm.ops(token_ids,
                            token_ids_reverse, embedding_weights)

        return output_tensor

class ElmoGru(object):

    '''
    构建tf计算图 for NLMs
    所有的超参都在config中
    '''

    def __init__(self, config):
        self.DTYPE = 'float32'
        self.DTYPE_INT = 'int64'
        self.config = config
        self.n_tokens_vocab = self.config['n_tokens_vocab']
        self.projection_dim = self.config['projection_dim']

        self.save_json_dir = self.config['save_json_dir']
        self.block_gru = blocks.BlockGRU(self.save_json_dir)


    def ops(self, token_ids, token_ids_reverse):
        # 通过word2vec 进行初始化embedding_weights
        emb_file = self.config.get("word_emb_file", "")
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(fname=emb_file)
        tmp_embd = np.zeros((self.n_tokens_vocab, self.projection_dim), dtype=self.DTYPE)
        words = w2v_model.vocab

        for i, word in enumerate(words):
            tmp_embd[i + 4][:] = w2v_model[word]
        print("load word2vec file for word embedding")

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            embedding_weights = tf.get_variable(
                "embedding", [self.n_tokens_vocab, self.projection_dim],
                dtype=self.DTYPE,
                initializer=tf.constant_initializer(tmp_embd),
                trainable=True
            )

            # self._build(token_ids, token_ids_reverse)
            output_tensor = self.block_gru.ops(token_ids,
                            token_ids_reverse, embedding_weights)

        return output_tensor


