#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################

"""
File: lang_tool.py
Author: wangyan(wangyan@aibot.me)
Date: 2018/09/24 11:53:24
Brief:  基础分词
"""

import os
import re
import sys
import jieba


class LangTool(object):
    def __init__(self):
        self.stopword_dic = set()
        self.trans2samp_dic = {}
        self.pattern_list = []

    def load(self, conf_dir):
        #wordseg
        file_userdict = conf_dir + "/seg_userdict.dic"
        jieba.load_userdict(file_userdict)

        #stopword
        file_stopword = conf_dir + "/stopword.dic"
        for line in open(file_stopword, encoding = "utf-8"):
            self.stopword_dic.add(line.strip())

        #load
        file_trans2samp = conf_dir + "/trans2samp.dic"
        for line in open(file_trans2samp, encoding = "utf-8"):
            items = line.strip().split("\t")
            if len(items) != 2:
                continue
            self.trans2samp_dic[items[0]] = items[1]
        
        # pattern
        self.pattern_list.append(("mail", re.compile("[0-9a-zA-Z\.]+@[\d\w]+\.[\w+]+")))
        self.pattern_list.append(("telephone", re.compile("(?:1[3|4|5|7|8][0-9](?:[0-9]{8}))(?=[^0-9]|$)")))
        self.pattern_list.append(("telephone", re.compile("((?:[\(（]\s*\d{3,4}\s*[\)）]([-\s])?)|(?:(\d{3,4})[-\s]))\d{7,8}(?=[^0-9]|$)")))
        return True
    

    def seg(self, query):
        tok_list = []
        for x in jieba.cut(query):
            word = str(x)
            tok_list.append(word.lower())
        return tok_list
    
    def trans2samp(self, query):
        # 繁简转换
        ch_list = []
        for ch in query.strip().lower():
            ch_list.append(self.trans2samp_dic.get(ch, ch))
        return "".join(ch_list)

    def norm_tok(self, tok_list = []):
        result = []
        for word in tok_list:
            # 电话 & 邮箱归一化
            for tag, pat in self.pattern_list:
                if pat.match(word):
                    # print("input->", query, "tag=",tag)
                    result.append(tag)
                else:
                    result.append(word)
        return result


