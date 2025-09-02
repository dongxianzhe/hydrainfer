#!/bin/bash 

source ../common.sh

REQUEST_RATES=10
NUM_REQUESTS=3
host="127.0.0.1"
port="8891"

export CUDA_VISIBLE_DEVICES=1
declare -A MODELS=(
    ["llava-hf/llava-1.5-7b-hf"]="/models/llava-1.5-7b-hf"
    # ["llava-hf/llava-v1.6-vicuna-7b-hf"]="/models/llava-v1.6-vicuna-7b-hf"
    # ["Qwen/Qwen2-VL-7B"]="/models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024"
    # ["deepseek-ai/deepseek-vl2-tiny"]="/models/deepseek-vl2-tiny"
)

for MODEL in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL]}"
    
    echo "=============================================="
    echo "Evaluating model: $MODEL"
    echo "Path: $MODEL_PATH"
    echo "=============================================="
    
    echo "starting api server"
    RAY_DEDUP_LOGS=0 \
        conda run -n hydrainfer --no-capture-output \
        python -m hydrainfer.entrypoint.entrypoint \
        model.path=$MODEL_PATH \
        cluster.epdnode.executor.use_flash_infer=false \
        cluster.epdnode.executor.multi_streams_forward=false \
        cluster.epdnode.executor.multi_threads_forward=false \
        cluster=single \
        apiserver.host=$host \
        apiserver.port=$port \
        ignore_eos=true \
        > $RESULT_PATH/${MODEL//\//-}-api_server.log 2>&1 &

    wait_api_server $host $port

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
        --backend=ours \
        --textcaps=1

    clean_up
    sleep 10
done