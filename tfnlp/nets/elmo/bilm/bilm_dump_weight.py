#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: bilm_test.py
Author: wangyan 
Date: 2019/02/13 11:53:24
Brief: 语言模型 dump
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from tfnlp.nets.elmo.bilm import bilm_core


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--outfile', help='Output hdf5 file with weights')

    args = parser.parse_args()
    bilm_core.dump_weights(args.save_dir, args.outfile)

