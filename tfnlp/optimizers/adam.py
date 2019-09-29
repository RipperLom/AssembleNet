#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: lamb.py
Author: wangyan 
Date: 2018/11/29 11:53:24
Brief: adam weight decay优化算法
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import sys
import logging

import tensorflow as tf



class AdamOptimizer(object):
    """
    a optimizer class: AdamOptimizer
    """

    def __init__(self, config=None):
        """
        init function
        """
        self.lr = float(config["learning_rate"])

    def ops(self):
        """
        operation
        """
        return tf.train.AdamOptimizer(learning_rate=self.lr)



