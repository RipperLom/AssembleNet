#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: split_train.py
Author: wangyan 
Date: 2019/09/05 11:53:24
Brief:  切分训练数据
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys


def split_train(max_num = 100000):
    i = 0
    file_id = 0
    out_name =  "train_part_%05d.txt" %(file_id)
    print("split data to file=[%s]" % (out_name))
    out_file = open(out_name, "w")
    for line in sys.stdin:
        if i >= max_num:
            out_file.close()
            file_id += 1
            out_name =  "train_part_%05d.txt" % (file_id)
            print("split data to file=[%s]" % (out_name))
            out_file = open(out_name, "w")
            i = 0
        i += 1
        out_file.write(line)
    out_file.close()
    return True


if __name__ == "__main__":
    max_num = int(sys.argv[1])
    split_train(max_num)

