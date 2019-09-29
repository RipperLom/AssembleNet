#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 

"""
File: vocab.py
Author: wangyan 
Date: 2019/07/17 11:53:24
Brief:  词典加载
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import glob
import logging
import collections
import numpy as np

from tfnlp.utils import text_norm



class VocabBase(object):
    """
        基本词典word<->id 转换
    """
    def __init__(self, vocab_file):
        self.unk = "[UNK]"
        self.name2id_dic = self.load_vocab(vocab_file)
        self.id2name_dic = {v: k for k, v in self.name2id_dic.items()}

    def get_vocab_list(self):
        return list(self.name2id_dic.keys())

    def load_vocab(self, vocab_file):
        '''
            Loads a vocabulary file into a dictionary
        '''
        idx = 0
        vocab = collections.OrderedDict()
        with open(vocab_file, "r") as reader:
            while True:
                token = text_norm.convert_to_unicode(reader.readline())
                if not token:
                    break
                items = token.strip().split("\t")
                if len(items) > 2:
                    continue
                token = items[0]
                index = int(items[1]) if len(items) == 2 else idx
                vocab[token] = index
                idx += 1

                # print(token, index)
        return vocab

    def keys(self):
        return self.name2id_dic.keys()
    
    def tokens2ids(self, tokens):
        return self.convert_by_vocab(self.name2id_dic, tokens)

    def ids2tokens(self, ids):
        return self.convert_by_vocab(self.id2name_dic, ids)
    
    def convert_by_vocab(self, vocab = {}, items = []): 
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            if item in vocab:
                output.append(vocab[item])
            else:
                output.append(vocab[self.unk])
        return output


class Vocabulary(object):
    '''
    基本的word <-> id 相互映射
    '''
    def __init__(self, file_name, check = False):
        '''
        filename = the vocabulary file.  It is a flat text file with one
            (normalized) token per line.  In addition, the file should also
            contain the special tokens <S>, </S>, <UNK> (case sensitive).
        '''
        self.id2word = []
        self.word2id = {}
        self.unk = -1
        self.bos = -1
        self.eos = -1
        if not self._load_dic(file_name, check):
            logging.error("load file=[%s] error!" % (file_name))
            return None

    def _load_dic(self, file_name, check = False):
        if not os.path.isfile(file_name):
            logging.warn("the file=[%s] not exist! " % (file_name))
            return False
        
        with open(file_name, encoding="utf-8") as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == '<S>':
                    self.bos = idx
                elif word_name == '</S>':
                    self.eos = idx
                elif word_name == '<UNK>':
                    self.unk = idx
                if word_name == '!!!MAXTERMID':
                    continue
                
                self.id2word.append(word_name)
                self.word2id[word_name] = idx
                idx += 1
        # check to ensure file has special tokens
        if check:
            if self.bos == -1 or self.eos == -1 or self.unk == -1:
                raise ValueError("Ensure the vocabulary file has "
                                 "<S>, </S>, <UNK> tokens")
        print("load vocab_num = [%d] success!" % (idx))
        return True
    

    def decode(self, cur_ids):
        '''
        Convert a list of ids to a sentence, with space inserted.
        '''
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence, reverse=False, split=True):
        '''
        Convert a sentence to a list of ids, with special tokens added.
        Sentence is a single string with tokens separated by whitespace.

        If reverse, then the sentence is assumed to be reversed, and
            this method will swap the BOS/EOS tokens appropriately.
        '''
        if split:
            word_ids = [
                self.word_to_id(cur_word) for cur_word in sentence.split()
            ]
        else:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        if reverse:
            return np.array([self.eos] + word_ids + [self.bos], dtype=np.int32)
        else:
            return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)

    def word_to_id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        return self.unk

    def id_to_word(self, cur_id):
        return self.id2word[cur_id]

    @property
    def size(self):
        return len(self.id2word)


