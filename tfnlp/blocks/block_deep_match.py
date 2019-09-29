#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/9/13 5:51 PM
# software: PyCharm

import tensorflow as tf


class BlockDeepMatch(object):
    '''
    文本匹配的分类问题，concat [v1, v2, v1 - v2, v1 * v2]，进行全联接分类。
    '''
    def __init__(self, config = {}):
        self.name = config.get('name', 'deep_match_block')
        self.n_class = config.get('n_class', 2)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.drop_rate = config.get('drop_rate', 0)
        self.load_state = False
    
    def load(self, weight_file):
        self.load_state = True
        return True

    def dump(self, weight_file):
        return True

    def ops(self, input_left, input_right):
        '''
        an op to compute diff between two sentence representations.
        Args:
            input_left: tensor of sentence representation, 左侧句子表示
            input_right: tensor of sentence representation, 右侧句子表示
        Returns:
            output_tensor: 两个句子表示的相似程度，分类问题
        Raises:
            None
        '''
        if not self.load_state:
            ## Combinations
            input_diff = input_left - input_right
            input_mul = input_left * input_right

            ### MLP
            mlp_input = tf.concat([input_left, input_right, input_diff, input_mul], 1)
            h_1 = tf.layers.dense(mlp_input, self.hidden_dim)
            h_1 = tf.nn.dropout(h_1, 1 - self.drop_rate)
            h_2 = tf.layers.dense(h_1, self.hidden_dim)
            h_2 = tf.nn.dropout(h_2, 1 - self.drop_rate)
            h_3 = tf.layers.dense(h_2, self.hidden_dim)
            h_3 = tf.nn.dropout(h_3, 1 - self.drop_rate)

            # Get prediction
            output_tensor = tf.layers.dense(h_3, self.n_class)
            output_tensor = tf.nn.dropout(output_tensor, keep_prob=1 - self.drop_rate)
            return output_tensor
        else:
            pass
