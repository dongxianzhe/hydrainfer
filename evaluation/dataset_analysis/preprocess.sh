#!/bin/bash 

source ../common.sh

model="llava-hf/llava-1.5-7b-hf"
model_path="/models/llava-1.5-7b-hf"
declare -A dataset_to_path=(
    ["lmms-lab/MME"]="/datasets/lmms-lab/MME"
    ["lmms-lab/TextCaps"]="/datasets/lmms-lab/TextCaps"
    ["lmms-lab/POPE"]="/datasets/lmms-lab/POPE"
    ["lmms-lab/textvqa"]="/datasets/lmms-lab/textvqa"
    ["lmms-lab/VizWiz-VQA"]="/datasets/lmms-lab/VizWiz-VQA"
)

for dataset in "${!dataset_to_path[@]}"; do
    dataset_path="${dataset_to_path[$dataset]}"
    echo ${dataset} ${dataset_path}
    conda run -n vllm --no-capture-output \
        python ${OUR_ROOT_PATH}/benchmark/data_preprocess.py \
            --dataset=${dataset} \
            --dataset-path=${dataset_path} \
            --model=${model} \
            --model-path=${model_path}
done

echo "all dataset preprocess finished"