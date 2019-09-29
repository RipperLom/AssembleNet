#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: option.py
Author: wangyan 
Date: 2019/04/11 11:53:24
Brief: 模型超参
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import six
import json

import argparse
import logging



class OptionBase(object):
    def __init__(self, config_path):
        self._config_dict = None
    
    def _parse(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf8') as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing Ernie model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            logging.info('%s: %s' % (arg, value))
        logging.info('------------------------------------------------')


