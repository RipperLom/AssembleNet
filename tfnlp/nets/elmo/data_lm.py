#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2018 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: data_lm.py
Author: wangyan 
Date: 2018/01/28 11:53:24
Brief: 语言模型加载
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import glob
import random
import logging
import numpy as np

from tfnlp.nets.elmo import vocab


class LMDataSet(object):
    """
    语言模型数据
    数据：多个文件
         每个文件，每行一个句子，词用\t分割
    """
    def __init__(self, file_pat, vocab, reverse=False, test=False, shuffle=False):
        """
        Args:
            file_pat: 文件名字模板
            vocab: 词典对象
            reverse: 正向迭代还是逆向迭代
            test： True 从前到后迭代一轮就stop
            shuffle: 打乱句子
        """
        self._vocab = vocab
        self._all_shards = glob.glob(file_pat)
        self._shard_to_choose = []
        self._reverse = reverse
        self._flag_test = test
        self._shuffle = shuffle

        self._use_char_inputs = hasattr(self._vocab, "encode_chars")

        #tmp for shard
        self._i = 0
        self._nids = 0
        self._ids_list = []
        print("find %d shards at %s" % (len(self._all_shards), file_pat))

    def get_sents(self):
        """
        读取一个句子
        """
        while True:
            if self._i == self._nids:
                self._ids_list = self._load_random_shard()
            ret = self._ids_list[self._i]
            self._i += 1
            yield ret

    def iter_batches(self, batch_size, num_steps):
        """
        batch 读取句子,
        批次大小 * 窗口大小

        x= w(1), w(2), w(num_steps)
        y = w(1 + 1), w(2 + 1), w(num_steps + 1)
        """
        max_word_len = self.max_word_length

        cur_stream = [None] * batch_size
        no_more_data = False

        while True:
            #define X , Y
            inputs = np.zeros([batch_size, num_steps], np.int32)
            if max_word_len is not None:
                char_inputs = np.zeros([batch_size, num_steps, max_word_len], np.int32)
            else:
                char_inputs = None
            targets = np.zeros([batch_size, num_steps], np.int32)

            for i in range(batch_size):
                cur_pos = 0
                while cur_pos < num_steps:
                    if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
                        try:
                            cur_stream[i] = list(next(self.get_sents()))
                        except StopIteration:
                            # No more data, exhaust current streams and quit
                            no_more_data = True
                            break
                    
                    how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
                    next_pos = cur_pos + how_many

                    inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
                    if max_word_len is not None:
                        char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][
                                                                        :how_many]
                    targets[i, cur_pos:next_pos] = cur_stream[i][0][1:how_many+1]

                    cur_pos = next_pos
                    cur_stream[i][0] = cur_stream[i][0][how_many:]
                    if max_word_len is not None:
                        cur_stream[i][1] = cur_stream[i][1][how_many:]

            if no_more_data:
                # There is no more data.  Note: this will not return data
                # for the incomplete batch
                break

            X = {'token_ids': inputs, 'tokens_characters': char_inputs,
                    'next_token_id': targets}
            yield X

    
    def _load_random_shard(self):
        """
        随机选择一个文件，并且read it
        """
        if self._flag_test:
            if len(self._all_shards) == 0:
                raise StopIteration
            else:
                shard_name = self._all_shards.pop()
        else:
            #随机选择文件名
            if len(self._shard_to_choose) == 0:
                self._shard_to_choose = list(self._all_shards)
                random.shuffle(self._shard_to_choose)
            shard_name = self._shard_to_choose.pop()

        shard_ids = self._load_shard(shard_name)
        self._i = 0
        self._nids = len(shard_ids)
        return shard_ids


    def _load_shard(self, shard_name):
        """Read one file and convert to ids.
        Args:
            shard_name: file path.
        Returns:
            list of (id, char_id) tuples.
        """
        print("load data from file=%s ..." % (shard_name))
        #load raw data
        sents_raw = []
        with open(shard_name, encoding= "utf-8") as infile:
            sents_raw = infile.readlines()
        if self._shuffle:
            random.shuffle(sents_raw)
        
        #encode all sents
        result = []
        for line in sents_raw:
            result.append(self.sents2_vec(line))
        return result

    def sents2_vec(self, text):
        """
            把\t分割的句子变成 ids_list 和 字符_list
        Args:
            text: \t 分割的文本    
        Returns:
            (ids, char_ids)
        """
        sents = text.split()
        if self._reverse:
            sents.reverse()
        
        sents_ids = self._vocab.encode(sents, self._reverse, split=False)
        sents_char_ids = [None] * len(sents_ids)
        if self._use_char_inputs:
            sents_char_ids = self._vocab.encode_chars(sents, self._reverse, split=False)
        return (sents_ids, sents_char_ids)
    
    @property
    def max_word_length(self):
        if self._use_char_inputs:
            return self._vocab.max_word_length
        else:
            return None
    
    @property
    def vocab(self):
        return self._vocab



class BiLMDataset(object):
    '''
    双向语言模型数据
    '''
    def __init__(self, file_pat, vocab, test=False, shuffle=False):
        '''
        bidirectional version of LMDataset
        '''
        self._data_forward = LMDataSet(file_pat, vocab, False, test, shuffle)
        self._data_reverse = LMDataSet(file_pat, vocab, True, test, shuffle)

    def iter_batches(self, batch_size, num_steps):
        '''
        双向语言模型迭代输入
        '''
        for X, Xr in zip(
            self._data_forward.iter_batches(batch_size, num_steps),
            self._data_reverse.iter_batches(batch_size, num_steps)
            ):

            for k, v in Xr.items():
                X[k + '_reverse'] = v
            yield X



##### list of sents to inputs  #####

class TokenBatcher(object):
    ''' 
    多个句子 转换成 词的id矩阵
    '''
    def __init__(self, lm_vocab_file):
        '''
        lm_vocab_file = the language model vocabulary file (one line per
            token)
        '''
        self._lm_vocab = vocab.Vocabulary(lm_vocab_file)


    def batch_sents(self, sentences):
        '''
        Batch the sentences as word ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        type: List[List[str]]
        sentences : [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) + 2

        X_ids = np.zeros((n_sentences, max_length), dtype=np.int64)

        for k, sent in enumerate(sentences):
            length = len(sent) + 2
            ids_without_mask = self._lm_vocab.encode(sent, split=False)
            # add one so that 0 is the mask value
            X_ids[k, :length] = ids_without_mask + 1
        return X_ids



class Batcher(object):
    '''
    Batch sentences of tokenized text into character id matrices.
    '''
    def __init__(self, lm_vocab_file, max_token_length):
        '''
        lm_vocab_file = 词典文件 
        max_token_length = 每个tok 最大ch 个数
        '''
        self._lm_vocab = vocab.UnicodeCharsVocabulary(
            lm_vocab_file, max_token_length
        )
        self._max_token_length = max_token_length

    def batch_sents(self, sentences):
        '''
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        eg. [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) + 2

        X_char_ids = np.zeros(
            (n_sentences, max_length, self._max_token_length),
            dtype=np.int64
        )

        for k, sent in enumerate(sentences):
            length = len(sent) + 2
            char_ids_without_mask = self._lm_vocab.encode_chars(
                sent, split=False)
            # add one so that 0 is the mask value
            X_char_ids[k, :length, :] = char_ids_without_mask + 1

        return X_char_ids


class InvalidNumberOfCharacters(Exception):
    pass

