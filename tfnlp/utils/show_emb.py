#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: show_emb.py
Author: wangyan 
Date: 2019/04/11 11:53:24
Brief: tsne 降为显示emb
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import argparse
import logging

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class ShowEmb(object):
    def __init__(self):
        pass
    

    def load(self, file_name):
        """
            load emb file from txt file
        """

        return True


    def show(self, pic_name):

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500

        # low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        # labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        # plot_with_labels(low_dim_embs, labels, pic_name)
        return True


    def plot_with_labels(self, low_dim_embs, labels, filename):
        """
            draw visualization of distance between embeddings
        """
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')
        plt.savefig(filename)



