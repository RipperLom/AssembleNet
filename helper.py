#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/9/16 9:56 PM
# software: PyCharm


import os
import argparse

__all__ = ['get_args_parser', 'get_args']


def get_args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='train',
                        help='task: train/test, the default value is train.')
    # parser.add_argument('--task_py', default='./task/elmo_emotion/elmo_emotion_task',
    parser.add_argument('--task_py', default='./task/elmo_emotion/elmo_emotion_task',
                        help='entrance of task.')
    parser.add_argument('--task_class', default='ElmoEmotionTask',
                        help='Class of task.')
    parser.add_argument('--conf_path', default='./task/elmo_emotion/elmo_gru_emotion_listwise.json',
                        help='conf_path.')
    return parser.parse_args()


def get_args():

    args = get_args_parser()
    return args

'''
--task
train
--task_py
./task/elmo_emotion/elmo_emotion_task
--task_class
ElmoEmotionTask
--conf_path
./task/elmo_emotion/elmo_lstm_emotion_listwise.json
'''
