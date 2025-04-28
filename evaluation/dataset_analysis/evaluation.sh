#!/bin/bash 

source ../common.sh

MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/llava-v1.6-vicuna-7b-hf"
datasets=("lmms-lab/MME" "lmms-lab/TextCaps" "lmms-lab/POPE" "lmms-lab/textvqa" "lmms-lab/VizWiz-VQA")

for i in {0..4}; do
    CUDA_VISIBLE_DEVICES=$i conda run -n vllm --no-capture-output python $OUR_ROOT_PATH/benchmark/data_preprocess.py --model-path=$MODEL_PATH --dataset="${datasets[$i]}" &
done

wait

echo "all dataset preprocess finished"