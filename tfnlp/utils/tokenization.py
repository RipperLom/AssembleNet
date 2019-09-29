#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################

"""
File: tokenization.py
Author: wangyan 
Date: 2018/11/29 11:53:24
Brief: 在bert的 toknization 上进行修改
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import re
import six
import unicodedata
import collections

from tfnlp.utils import vocab
from tfnlp.utils import text_norm


class BasicTokenizer(object):
    """
        Runs basic tokenization (punctuation splitting, lower casing, etc.)
    """
    def __init__(self, do_lower_case = True):
        """Constructs a BasicTokenizer.
        Args:
            do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.char_type = text_norm.CharType()

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = text_norm.convert_to_unicode(text)
        text = self._clean_text(text)
        
        # 中文前后加空格
        text = self._tokenize_chinese_chars(text)
        orig_tokens = text_norm.whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = text_norm.whitespace_tokenize(" ".join(split_tokens))
        return output_tokens
    
    def text_norm(self, text):
        """
        norm text 
        """
        text = text_norm.convert_to_unicode(text)
        text = self._clean_text(text)
        orig_tokens = text_norm.whitespace_tokenize(text)
        tok_list = []
        for tok in orig_tokens:
            tok = self._run_strip_accents(tok)
            tok_list.append(tok)
        return " ".join(tok_list)
    
    
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        output = []
        start_new_word = True
        while i < len(chars):
            char = chars[i]
            if self.char_type.is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        ## 中文CJK 前后加空格
        output = []
        for char in text:
            cp = ord(char)
            if self.char_type.is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _clean_text(self, text):
        # 删除控制字符 + 归一化空白字符
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self.char_type.is_control(char):
                continue
            if self.char_type.is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """
        Runs WordPiece tokenziation.
    """
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        
        For example:
        input = "unaffable"
        output = ["un", "##aff", "##able"]

        Args:
        text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.
        Returns:
        A list of wordpiece tokens.
        """
        text = text_norm.convert_to_unicode(text)
        output_tokens = []
        for token in text_norm.whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start : end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens



class FullTokenizer(object):
    """Runs end-to-end tokenziation."""
    def __init__(self, vocab_file, do_lower_case = True, do_sub_tok = False):
        self.do_lower = do_lower_case
        self.do_sub_tok = do_sub_tok

        self.vocab = vocab.VocabBase(vocab_file)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab = self.vocab.name2id_dic)
        self.char_type = text_norm.CharType()

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            # print("token-->", token)
            if self.do_sub_tok: 
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
            else:
                split_tokens.append(token)
        return split_tokens

    def tokens2ids(self, tokens):
        return self.vocab.tokens2ids(tokens)

    def ids2tokens(self, ids):
        return self.vocab.ids2tokens(ids)

    def get_vocab_words(self):
        return self.vocab.get_vocab_list()


