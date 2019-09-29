#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/9/10 9:01 PM
# software: PyCharm

import tensorflow as tf

class BlockEmbedding(object):
    DTYPE = 'float32'
    def __init__(self, config={}):
        self.name = config.get('name', 'block_embedding')
        self.n_vocabs = config.get('n_vocabs', 0)
        self.hidden_dim = config.get('hidden_dim', 0)

    def load(self, weight_file):
        '''
        an op to load weight file initializing the tensor
        Args:
            weight_file: path of h5 weights
        Returns:
            result of loading
        Raises:
            None
        '''
        with open(weight_file, 'r', encoding='utf-8') as f:
            self.embedding_table_ = f['embedding_table']
            self.n_vocabs, self.hidden_dim = self.embedding_table_.shape
            self.embedding_weights = tf.get_variable("embedding", [self.n_vocabs, self.hidden_dim],
                dtype=self.DTYPE, initializer=tf.constant_initializer(self.embedding_table_), trainable=True)
        return True

    def dump(self, weight_file):
        return True

    def ops(self, input_tensor):
        '''
        an op to embedding_lookup
        Args:
            input_tensor: tensor [batchSize, seqLen]
        Returns:
            output_tensor: tensor [batchSize, seqLen, hidden_dim]
        Raises:
            None
        '''
        output_tensor = tf.nn.embedding_lookup(self.embedding_weights, input_tensor,
                                               name='embedding_lookup')
        return output_tensor


