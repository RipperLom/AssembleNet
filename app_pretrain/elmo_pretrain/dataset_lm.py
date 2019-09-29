#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/9/25 3:13 PM
# software: PyCharm

import sys

from tfnlp.dataset.data_lm import DateSetBlm


def usage():
    """
    usage
    """
    print(sys.argv[0], "options")
    print("options")
    print("\tvocab_file: vocab file path")
    print("\tinput_file: input seg file path")
    print("\tout_file: output tf_record file path")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        usage()
        sys.exit(-1)

    vocab_file = sys.argv[1]
    input_file = sys.argv[2]
    out_file = sys.argv[3]

    dataset = DateSetBlm()
    dataset.load_vocab(vocab_file)
    dataset.tf_example_file(input_file, out_file)
