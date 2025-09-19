#!/bin/bash 
source ../common.sh

############################## params ##############################
REQUEST_RATES=10
NUM_REQUESTS=3
host="127.0.0.1"
port="8891"
start_server_max_retry=5
find_free_gpus_max_retry=1000
declare -A MODELS=(
    ["llava-hf/llava-1.5-7b-hf"]="/models/llava-1.5-7b-hf"
    # ["llava-hf/llava-v1.6-vicuna-7b-hf"]="/models/llava-v1.6-vicuna-7b-hf"
    # ["Qwen/Qwen2-VL-7B"]="/models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024"
    # ["deepseek-ai/deepseek-vl2-tiny"]="/models/deepseek-vl2-tiny"
    ["OpenGVLab/InternVL2-26B"]="/models/OpenGVLab/InternVL2-26B"
)
gpu_configs=(
    1
    # 2
    # 4
    # 8
)
additional_server_configs=(
    ""
)
####################################################################

start_server(){
    echo "starting api server"
    attempt=0
    while [ $attempt -lt $start_server_max_retry ]; do
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        RAY_DEDUP_LOGS=0 \
            conda run -n hydrainfer --no-capture-output \
            python -m hydrainfer.entrypoint.entrypoint \
            model.path=$MODEL_PATH \
            apiserver.host=$host \
            apiserver.port=$port \
            ignore_eos=true \
            $additional_server_config \
            > $RESULT_PATH/${log_prefix}-api_server.log 2>&1 &
        pid=$!
        if wait_api_server $host $port $pid; then
            echo "server start success."
            break
        else
            echo "server start failed. Retrying..."
            clean_up
            attempt=$((attempt + 1))
        fi
    done
}

send_requests(){
    echo "start to send requests"
    conda run -n hydrainfer --no-capture-output \
        python ${OUR_ROOT_PATH}/benchmark/benchmark.py \
        --num-requests=$NUM_REQUESTS \
        --model=${MODEL} \
        --model-path=${MODEL_PATH} \
        --host $host \
        --port $port \
        --result-path=$RESULT_PATH/${log_prefix}-result.json \
        --request-rate $REQUEST_RATES \
        --show-result=4 \
        --backend=ours \
        --textcaps=1
}

test_correctness(){
    start_server
    send_requests

    clean_up
    sleep 10
}

for number_gpus_need in ${gpu_configs[@]}; do
    for MODEL in "${!MODELS[@]}"; do
        MODEL_PATH="${MODELS[$MODEL]}"
        for additional_server_config in "${additional_server_configs[@]}"; do
            attempt=0
            while [ $attempt -lt $find_free_gpus_max_retry ]; do
                read -a free_gpus <<< "$(get_free_gpus)"
                selected_gpus=("${free_gpus[@]:0:$number_gpus_need}")
                gpu_list=$(IFS=,; echo "${selected_gpus[*]}")
                CUDA_VISIBLE_DEVICES="$gpu_list"
                echo "=============================================="
                echo "Evaluating model: $MODEL"
                echo "Path: $MODEL_PATH"
                echo "additional_server_config: $additional_server_config"
                echo "free gpus ${free_gpus[@]}"
                echo "number_gpus_need ${number_gpus_need}"
                echo "set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
                echo "=============================================="
                safe_gpus_info="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
                safe_additional_server_config=$(echo "$additional_server_config" | sed 's/[\/:*?"<>|&!(){}]/_/g')
                safe_model=$(echo "$MODEL" | sed 's/[\/:*?"<>|&!(){}]/_/g')
                log_prefix="${safe_gpus_info}-${safe_additional_server_config}-${safe_model}"
                if [ ${#free_gpus[@]} -ge "$number_gpus_need" ]; then
                    test_correctness
                    break
                else
                    echo "not enough available gpus. retrying..."
                    clean_up
                    sleep 60
                    attempt=$((attempt + 1))
                fi
            done
        done
    done
done