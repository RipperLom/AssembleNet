#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/9/24 11:44 AM
# software: PyCharm
# Brief: Ada Grad 自适应学习率


from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import sys
import logging

import tensorflow as tf


class AdagradOptimizer(object):
    """
    a optimizer class: AdamOptimizer
    """

    def __init__(self, config=None):
        """
        init function
        """
        self.lr = float(config["learning_rate"])
        self.init_accu = float(config['initial_accumulator_value'])

    def ops(self):
        """
        operation
        """
        return tf.train.AdagradOptimizer(learning_rate=self.lr,
                initial_accumulator_value=self.init_accu)

