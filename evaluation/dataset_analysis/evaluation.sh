#!/bin/bash 

export CUDA_VISIBLE_DEVICES=1
export MODEL_PATH=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")

RESULT_DIR="$SCRIPT_PATH/result"
mkdir -p $RESULT_DIR

cd ../../benchmark

DATASETS=("mme" "textcaps" "vega")

for DATASET in "${DATASETS[@]}"; do
    echo "Start analysis dataset $DATASET"

    CMD="python analysis_dataset.py --dataset $DATASET"
    
    if [ -n "$MODEL_PATH" ]; then
        CMD="$CMD --model $MODEL_PATH"
    fi

    $CMD > $RESULT_DIR/${DATASET}.log

    sleep 3
done
