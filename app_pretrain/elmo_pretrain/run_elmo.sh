#! /bin/bash

set -x

# split train

ROOT_PATH=/Users/admin/Desktop/git/oceanus-bot-model
TASK_PATH=${ROOT_PATH}/task/elmo_pretrain

cd ${ROOT_PATH}

function task_elmo()
{
    # 数据转换
    #python ${TASK_PATH}/dataset_lm.py \
    #    ${TASK_PATH}/data/vocab.txt \
    #    ${TASK_PATH}/data/train.txt \
    #    ${TASK_PATH}/data/tf_record

    # train
    python ${ROOT_PATH}/assemble_net.py \
        -task train \
        -task_py ${TASK_PATH}/elmo_lstm_task \
        -task_class LstmElmoTask \
        -conf_path ${TASK_PATH}/conf/pre_train_elmo_2_lstm.json

    return 0
}

task_elmo
