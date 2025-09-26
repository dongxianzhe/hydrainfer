#!/bin/bash 

source ../common.sh

REQUEST_RATES=10
NUM_REQUESTS=3
host="127.0.0.1"
port="8891"

export CUDA_VISIBLE_DEVICES=1
declare -A MODELS=(
    ["llava-hf/llava-1.5-7b-hf"]="/models/llava-1.5-7b-hf"
    ["llava-hf/llava-v1.6-vicuna-7b-hf"]="/models/llava-v1.6-vicuna-7b-hf"
    ["Qwen/Qwen2-VL-7B"]="/models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024"
    ["deepseek-ai/deepseek-vl2-tiny"]="/models/deepseek-vl2-tiny"
)

start_server(){
    echo "starting api server"
    max_retry=5
    attempt=0
    while [ $attempt -lt $max_retry ]; do
        SCHEDULE_METHOD="${method}" \
        RAY_DEDUP_LOGS=0 \
            conda run -n hydrainfer --no-capture-output \
            python -m hydrainfer.entrypoint.entrypoint \
            model.path=$MODEL_PATH \
            apiserver.host=$host \
            apiserver.port=$port \
            ignore_eos=true \
            > $RESULT_PATH/${MODEL//\//-}-${method}-api_server.log 2>&1 &
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
        --result-path=${RESULT_PATH}/${MODEL//\//-}-result.json \
        --request-rate $REQUEST_RATES \
        --show-result=4 \
        --backend=ours \
        --textcaps=1
}

for MODEL in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL]}"
    
    echo "=============================================="
    echo "Evaluating model: $MODEL"
    echo "Path: $MODEL_PATH"
    echo "=============================================="
    
    start_server
    send_requests

    clean_up
    sleep 10
done