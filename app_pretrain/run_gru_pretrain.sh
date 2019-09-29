#! /bin/bash

set -x

ROOT_PATH=/Users/admin/Desktop/git/oceanus-bot-model
DATA_PATH=${ROOT_PATH}/data
TASK_PATH=${ROOT_PATH}/app_pretrain/elmo_pretrain

cd ${ROOT_PATH}

# 数据格式转换
function fmt_train()
{
    python ${ROOT_PATH}/app_pretrain/gen_lm_data.py \
        --conf_path=${DATA_PATH} \
        --input_file=${DATA_PATH}/pretrain/train.txt \
        --token_file=${DATA_PATH}/pretrain/train_tok.txt \
        --tf_file=${DATA_PATH}/pretrain/train_tfrecord
    return 0
}


function train_model()
{
    # 模型训练
    #python ${ROOT_PATH}/assemble_net.py \
    #    --task train \
    #    --task_py ${TASK_PATH}/elmo_gru_task \
    #    --task_class GruElmoTask \
    #    --conf_path ${TASK_PATH}/conf/pre_train_elmo_1_gru.json

    # dump weight
    python ${ROOT_PATH}/app_pretrain/dump_lm_data.py \
        --block_py ${ROOT_PATH}/tfnlp/blocks/block_gru \
        --block_class BlockGRU \
        --config_dir ${ROOT_PATH}/model/pre_train_elmo_1_gru/options.json \
        --ckpt_file ${ROOT_PATH}/model/pre_train_elmo_1_gru/1LayerGru.epoch1 \
        --outfile_weights ${ROOT_PATH}/model/pre_train_elmo_1_gru/weights.hdf5

    return 0
}

function fine_tune_task()
{
    python ${ROOT_PATH}/assemble_net.py \
        --task train \
        --task_py ${ROOT_PATH}/task/elmo_emotion/elmo_emotion_task \
        --task_class ElmoEmotionTask \
        --conf_path ${ROOT_PATH}/task/elmo_emotion/elmo_gru_emotion_listwise.json

    return 0
}

# fmt_train
train_model
# fine_tune_task
