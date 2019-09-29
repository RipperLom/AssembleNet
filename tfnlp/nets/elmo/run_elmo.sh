#! /bin/bash

set -x

WORK_DIR=/Users/wy/workspace/project/nlp/tfnlp
export PYTHONPATH=${WORK_DIR}

BIN_DIR=${WORK_DIR}/tfnlp/models/bilm


### train
python  ${BIN_DIR}/bilm_train.py \
        --save_dir  ./model/lm_model \
        --vocab_file  ${WORK_DIR}/data/vocab.txt \
        --train_prefix ${WORK_DIR}/data/train.txt

### dump weight
python ${BIN_DIR}/bilm_dump_weight.py \
    --save_dir ./model/lm_model \
    --outfile ./model/lm_model/lm_weights.hdf5


### test
python ${BIN_DIR}/bilm_test.py \
    --test_prefix ${WORK_DIR}/data/train.txt \
    --vocab_file ${WORK_DIR}/data/vocab.txt \
    --save_dir ./model/lm_model \
    --batch_size 2 


exit 0


