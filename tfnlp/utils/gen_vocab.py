#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 

"""
File: gen_vocab.py
Author: wangyan 
Date: 2019/09/25 11:53:24
Brief:  统计语料生成词典文件
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import glob
import logging
import collections
import numpy as np

from tfnlp.utils import text_norm

