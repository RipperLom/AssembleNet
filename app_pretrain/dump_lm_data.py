#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/9/25 6:14 PM
# software: PyCharm

import os
import sys
import logging
import warnings
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tfnlp.utils.common import import_object


class DumpLMData(object):

    def __init__(self):
        self.block = None
        pass


    def load(self, block_py, block_class, config_dir, ckpt_file, outfile_weights):
        '''
        check existence of files
        Args:
            block_py: block file
            block_class: block Class name
            ckpt_file: path of checkpoint
            config_dir: path of config
            outfile_weights: path of a HDF5 file
        Returns:
            True / False
        Raises:
            None
        '''
        if os.path.exists(config_dir):
            self.config_dir = config_dir
        else:
            logging.error("load config error!")
            return False

        parent_path = os.path.split(outfile_weights)[0]
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        self.outfile_weights = outfile_weights

        self.ckpt_file = ckpt_file

        self.block = import_object(block_py, block_class)(config_dir)
        return True


    def process(self):
        '''
        dump weights
        Args:
            None
        Returns:
            None
        Raises:
            None
        '''
        self.block.dump(self.ckpt_file, self.outfile_weights)

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_py', default='./tfnlp/blocks/block_lstm',
                        help='block file')
    parser.add_argument('--block_class', default='BlockLSTM',
                        help='block Class name')
    parser.add_argument('--config_dir', default='./model/pre_train_elmo_1_lstm/options.json',
                        help='the vocab file')
    parser.add_argument('--ckpt_file', default='./model/pre_train_elmo_1_lstm/1LayerLstm.epoch1',
                        help='train file')
    parser.add_argument('--outfile_weights', default='./model/pre_train_elmo_1_lstm/weights.hdf5',
                        help='train token file')
    return parser.parse_args()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    args = get_args_parser()
    print(args)

    dump_data = DumpLMData()
    load = dump_data.load(args.block_py, args.block_class,
                          args.config_dir, args.ckpt_file, args.outfile_weights)
    print(load)
    dump_data.process()

