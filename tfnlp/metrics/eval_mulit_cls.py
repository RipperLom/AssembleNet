#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: eval_mulit_cls.py
Author: wangyan 
Date: 2019/09/10 11:53:24
Brief: 多类问题中召回率、准确率、总的精确度、F-score
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys



def evaluation(label_col, pred_col):
    labels = {}
    result = 0
    correct = 0
    sample_num = 0

    #for row_num, line in enumerate(open("d:/log")):
    for row_num, line in enumerate(sys.stdin):
        line = line.strip()
        line_tokens = line.split()
        token_num = len(line_tokens)
        if token_num < 1:
            continue
        try:
           true_label = line_tokens[label_col-1]
           pred_label = line_tokens[pred_col-1]
           if true_label in labels:
               labels[true_label][0] += 1
           else:
               labels[true_label] = [1,0,0]
           
           if pred_label in labels:
               labels[pred_label][1] += 1
           else:
               labels[pred_label] = [0,1,0]
           
           if true_label == pred_label:
               labels[pred_label][2] += 1
               correct += 1
           sample_num += 1 
        except Exception as e:
            sys.stderr.write("line %d line=%s fmt error!\n" % (row_num+1, line))

    if len(labels) == 0 or sample_num == 0:
        sys.stderr.write("error: have no valid data.\n")
        return
    

    #dump 评估结果
    print("每类整体评估指标:")
    print("%20s%20s%20s%20s%20s" % ("类名","召回率","准确率","F值","AUC"))
    precision = 0.0
    recall = 0.0
    F_score = 0.0
    AUC = 0.0
    fpr = 0.0
    for key in labels:
        val = labels[key]
        if val[0] == 0:
            continue
        result += val[1]
        if val[1] == 0:
            precision = 0.0
        else:
            precision = val[2]*1.0 / val[1]
        if val[0] == 0:
            recall = 0.0
        else:
            recall = val[2]*1.0 / val[0]
        if sample_num - val[0] == 0:
            fpr = 0.0
        else:
            fpr = (val[1]-val[2])*1.0/(sample_num - val[0])
        AUC = (recall-fpr+1)*1.0/2
        if recall >=-1e-20 and recall <= 1e-20 or precision >=-1e-20 and precision <= 1e-20:
            F_score = 0
        else:
            F_score = 2 / (1/precision + 1/recall)

        print("%20s%18.2f%%%18.2f%%%18.3f%18.3f" % (key,recall*100,precision*100,F_score,AUC))

    print("\n整体评估指标:")
    print("有分类比例=%.3f" % (1.0*result/sample_num))
    print("有分类样本准确率=%.3f" % (1.0*correct/result))
    print("正确率=%.3f" % (1.0*correct/sample_num))



if __name__ == "__main__":
    if 3 != len(sys.argv):
        sys.stderr.write(" Usage: cat test_file | cmd  label_col predict_col > result")
        sys.exit(-1)
    
    label_col = int(sys.argv[1])
    predict_col = int(sys.argv[2])
    if (label_col <= 0 or predict_col <= 0):
        print("label_col or label_col error!")
        sys.exit(-1)
    evaluation(label_col, predict_col)

