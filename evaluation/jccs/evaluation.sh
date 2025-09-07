#!/bin/bash

source ../common.sh
REQUEST_RATES="1 2 3 4 5 6 7 8 9 10 11 12"
NUM_REQUESTS=30
host="127.0.0.1"
port="8891"
export CUDA_VISIBLE_DEVICES=1
declare -A MODELS=(
    ["llava-hf/llava-1.5-7b-hf"]="/models/llava-1.5-7b-hf"
    ["llava-hf/llava-v1.6-vicuna-7b-hf"]="/models/llava-v1.6-vicuna-7b-hf"
    ["Qwen/Qwen2-VL-7B"]="/models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024"
    ["deepseek-ai/deepseek-vl2-tiny"]="/models/deepseek-vl2-tiny"
)


scenarios=(
    "--textcaps=0 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=1"
    "--textcaps=1 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=0"
    "--textcaps=0 --pope=1 --mme=0 --text_vqa=0 --vizwiz_vqa=0"
    "--textcaps=0 --pope=0 --mme=1 --text_vqa=0 --vizwiz_vqa=0"
    "--textcaps=0 --pope=0 --mme=0 --text_vqa=1 --vizwiz_vqa=0"
    "--textcaps=1 --pope=1 --mme=1 --text_vqa=1 --vizwiz_vqa=1"
)

methods=(
    "STEP"
    "REQUEST"
    "PREFILL"
    "CONTINUOUS"
    "STALLFREE"
)

try_start_server(){
    max_retry=5
    attempt=0
    while [ $attempt -lt $max_retry ]; do
        SCHEDULE_METHOD="${method}" \
        RAY_DEDUP_LOGS=0 \
            conda run -n hydrainfer --no-capture-output \
            python -m hydrainfer.entrypoint.entrypoint \
            model.path=$MODEL_PATH \
            cluster.epdnode.executor.multi_streams_forward=false \
            cluster.epdnode.executor.multi_threads_forward=false \
            cluster=single \
            cluster.epdnode.batch_scheduler_profiler.tpot_slo=0.04 \
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

for MODEL in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL]}"
    for method in "${methods[@]}"; do
        echo "=============================================="
        echo "Evaluating model: $MODEL"
        echo "Path: $MODEL_PATH"
        echo "method: ${method}"
        echo "=============================================="
        echo "starting api server"

        try_start_server
        # sleep 10000

        echo "start to send requests"
        for scenario in "${scenarios[@]}"; do
            conda run -n hydrainfer --no-capture-output \
                python ${OUR_ROOT_PATH}/benchmark/benchmark.py \
                --num-requests=$NUM_REQUESTS \
                --model=${MODEL} \
                --model-path=${MODEL_PATH} \
                --host $host \
                --port $port \
                --result-path=${RESULT_PATH}/${MODEL//\//-}-${method}-${scenario// /_}-result.json \
                --request-rate $REQUEST_RATES \
                --backend=ours \
                --method-name=${method} \
                ${scenario}
        done
        clean_up
        sleep 10
    done
done