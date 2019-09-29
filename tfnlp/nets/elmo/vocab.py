#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2018 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: vocab.py
Author: wangyan 
Date: 2018/01/28 11:53:24
Brief: 词典加载
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import glob
import logging
import numpy as np



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
        self._id2word = []
        self._word2id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1
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
                line = line.strip()
                word_name = line
                items = line.split("\t")
                if len(items) == 2:
                    word_name = items[0]
                if word_name == '<S>':
                    self._bos = idx
                elif word_name == '</S>':
                    self._eos = idx
                elif word_name == '<UNK>':
                    self._unk = idx
                if word_name == '!!!MAXTERMID':
                    continue
                
                self._id2word.append(word_name)
                self._word2id[word_name] = idx
                idx += 1
        # check to ensure file has special tokens
        if check:
            if self._bos == -1 or self._eos == -1 or self._unk == -1:
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
        if word in self._word2id:
            return self._word2id[word]
        return self.unk

    def id_to_word(self, cur_id):
        return self._id2word[cur_id]

    @property
    def bos(self):
        return self._bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def size(self):
        return len(self._id2word)



class UnicodeCharsVocabulary(Vocabulary):
    """
        字符级别 lookup表
        Vocabulary containing character-level and word level information.
        Has a word vocabulary that is used to lookup word ids and
        a character id that is used to map words to arrays of character ids.
        The character ids are defined by ord(c) for c in word.encode('utf-8')
        This limits the total number of possible char ids to 256.
        To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.
    """
    def __init__(self, file_name, max_word_length, **kwargs):
        super(UnicodeCharsVocabulary, self).__init__(file_name, **kwargs)
        self._max_word_length = max_word_length

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bos_char = 256  # <begin sentence>
        self.eos_char = 257  # <end sentence>
        self.bow_char = 258  # <begin word>
        self.eow_char = 259  # <end word>
        self.pad_char = 260  # <padding>

        num_words = len(self._id2word)
        self._word_char_ids = np.zeros([num_words, max_word_length], dtype=np.int32)
        
        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([self.max_word_length], dtype=np.int32)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r
        
        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)
        for i, word in enumerate(self._id2word):
            self._word_char_ids[i] = self._convert_word2char_ids(word)

        self._word_char_ids[self.bos] = self.bos_chars
        self._word_char_ids[self.eos] = self.eos_chars
        # TODO: properly handle <UNK>


    def word_to_char_ids(self, word):
        '''
            词转 字list
        '''
        if word in self._word2id:
            return self._word_char_ids[self._word2id[word]]
        else:
            return self._convert_word2char_ids(word)
    
    def encode_chars(self, sentence, reverse=False, split=True):
        '''
            Encode the sentence as a white space delimited string of tokens.
        '''
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence]
        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [self.bos_chars])
        else:
            return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word2char_ids(self, word):
        '''
           词 + 开头结尾加指定符合，其他位置填充
        '''
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char
        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length-2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[len(word_encoded) + 1] = self.eow_char
        return code


