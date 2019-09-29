#! /bin/bash

set -x

ROOT_PATH=/home/work/boss_bot/oceanus-bot-model
DATA_PATH=${ROOT_PATH}/data

export PYTHONPATH=$PYTHONPATH:${ROOT_PATH}


# 数据格式转换
function fmt_train()
{
    # 生成数据
    python ${ROOT_PATH}/tfnlp/nets/bert/gen_pretrain_data.py \
        --seg_path=${ROOT_PATH}/task/kefu_qu/conf/nlpbase \
        --vocab_file=${DATA_PATH}/bert/vocab.txt \
        --input_file=${DATA_PATH}/sample_text*,${DATA_PATH}/sample_jd.txt \
        --output_prefix=${DATA_PATH}/bert/train_bert_tfrecord  \
        --output_num=2 \
        --do_lower_case=true \
        --do_whole_word_mask=true \
        --max_seq_length=64 \
        --max_predictions_per_seq=20 \
        --random_seed=12345 \
        --dupe_factor=5 \
        --masked_lm_prob=0.15 \
        --short_seq_prob=0.1
    return 0
}


function train_model()
{
    python ${ROOT_PATH}/tfnlp/nets/bert/run_pretrain.py \
        --bert_config_file=${DATA_PATH}/bert/bert_config.json \
        --input_file=${DATA_PATH}/bert/train_bert_tfrecord* \
        --output_dir=${DATA_PATH}/bert/logs \
        --max_seq_length=64 \
        --max_predictions_per_seq=20 \
        --do_train=True \
        --train_batch_size=32 \
        --num_train_steps=100000 \
        --num_warmup_steps=10000 \
        --save_checkpoints_steps=1000 \
        --iterations_per_loop=1000 \
        --use_tpu=False
    return 0
}


fmt_train
# train_model


# fine tuning

