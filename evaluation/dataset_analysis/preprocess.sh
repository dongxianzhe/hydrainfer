#!/bin/bash 

source ../common.sh

# MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/llava-v1.6-vicuna-7b-hf"
MODEL_PATH="/models/llava-1.5-7b-hf"
# datasets=("/datasets/lmms-lab/MME" "/datasets/lmms-lab/TextCaps" "/datasets/lmms-lab/POPE" "/datasets/lmms-lab/textvqa" "/datasets/lmms-lab/VizWiz-VQA")
datasets=("/datasets/lmms-lab/TextCaps")

for i in {0..0}; do
    CUDA_VISIBLE_DEVICES=$i conda run -n vllm --no-capture-output python $OUR_ROOT_PATH/benchmark/data_preprocess.py --model-path=$MODEL_PATH --dataset="${datasets[$i]}"
done

wait

echo "all dataset preprocess finished"