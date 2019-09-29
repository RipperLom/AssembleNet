
#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: text_tok.py
Author: wangyan 
Date: 2019/09/25 11:53:24
Brief: 
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import glob
import random
import logging

import sys
from tfnlp.utils.tokenization import BasicTokenizer


tokenizer = BasicTokenizer()
for line in sys.stdin:
    line = line.strip()
    tok_list = tokenizer.tokenize(line)
    print(" ".join(tok_list))


