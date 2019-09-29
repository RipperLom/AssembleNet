#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: pre_data_jd.py
Author: wangyan 
Date: 2019/09/27 11:53:24
Brief: 
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import re
import os
import sys

def split_jb(line):
    sents = re.split("(?:[：，。？；;?\s]+\d[、.])|(?:[。；！？])", line)
    return sents


def filter_jd():
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        items = line.split("\t")
        if len(items) != 31:
            continue
        title = items[1]
        jd_desc = items[23]
        
        data = split_jb(jd_desc)
        for sent in data:
            sent = sent.strip()
            if len(sent) == 0:
                continue
            print(sent)
        print("")
        # print(title)
    return True

if __name__ == "__main__":
    filter_jd()
    print("done")


