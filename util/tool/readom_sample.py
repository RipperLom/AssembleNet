#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################

"""
File: readom_sample.py
Author: wangyan
Date: 2019/05/09 11:53:24
Brief: 
"""
import os
import re
import sys
import random


def random_line(size):
    buf = []
    for line in sys.stdin:
        line =  line.strip()
        if len(buf) < size:
            buf.append(line)
        else:
            seed = random.randint(0,len(buf)-1)
            print(buf[seed])
            buf = []
    return True


if __name__ == '__main__':
    if (2 != len(sys.argv)):
        print("Usage:cat + file_name | cmd  size > result")
        sys.exit(1)
    extract_size = int(sys.argv[1])    
    random_line(extract_size)

