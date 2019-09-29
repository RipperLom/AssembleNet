#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: GZhao_zhaoguanzhi
# datetime: 2019/8/29 5:42 PM
# software: PyCharm

import os
import warnings

from helper import get_args
from tfnlp.utils.common import import_object

warnings.filterwarnings('ignore')
args = get_args()
if True:
    import sys

    param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
    print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))

# os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
task = import_object(args.task_py, args.task_class)()
task.load(args.conf_path)

if args.task == 'train':
    task.train()
elif args.task == 'test':
    task.test()
else:
    print(sys.stderr, 'task type error.')
