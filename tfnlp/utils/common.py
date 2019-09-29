#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: common.py
Author: wangyan 
Date: 2019/09/10 11:53:24
Brief: 通用工具
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import glob
import json
import logging
import traceback



def clazz(config, str1, str2):
    return import_object(config[str1], config[str2])


def load_config(config_file):
    """
    load config
    """
    with open(config_file, "r") as f:
        try:
            conf = json.load(f)
        except Exception:
            logging.error("load json file %s error" % config_file)
    conf_dict = {}
    unused = [conf_dict.update(conf[k]) for k in conf]
    logging.debug("\n".join(
        ["%s=%s" % (u, conf_dict[u]) for u in conf_dict]))
    return conf_dict


def read_file_iter(dir_name):
    """
    yield file name from dir
    """
    for root, dirnames, filenames in os.walk(dir_name):
        for filename in filenames:
            name = os.path.join(root, filename)
            yield name


def file_list(file_prefix, file_num, suffix = ""):
    """
    gen file list
    """
    name_list = []
    for i in range(file_num):
        name = "%s_%04d%s" % (file_prefix, i, suffix)
        name_list.append(name)
    return name_list


def get_cards():
    """
    get gpu cards number
    """
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


def get_all_files(train_data_file):
    """
    get all files
    """
    train_file = []
    train_path = train_data_file
    if os.path.isdir(train_path):
        data_parts = os.listdir(train_path)
        for part in data_parts:
            train_file.append(os.path.join(train_path, part))
    else:
        train_file.append(train_path)
    return train_file


def merge_config(config, *argv):
    """
    merge multiple configs
    """
    cf = {}
    cf.update(config)
    for d in argv:
        cf.update(d)
    return cf


def import_object(module_py, class_str):
    """
    string to class
    """
    mpath, mfile = os.path.split(module_py)
    sys.path.append(mpath)
    module=__import__(mfile)
    try:
        return getattr(module, class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                (class_str, traceback.format_exception(*sys.exc_info())))


