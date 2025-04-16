#!/bin/bash 

source ../common.sh

export CUDA_VISIBLE_DEVICES=1
MODEL="llava-hf/llava-1.5-7b-hf"
MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf"
REQUEST_RATES=10
NUM_REQUESTS=10
host="127.0.0.1"
port="8888"

echo "starting api server"
RAY_DEDUP_LOGS=0 \
    conda run -n dxz_dev --no-capture-output \
    python -m dxz.entrypoint.entrypoint \
    model.name=$MODEL \
    model.path=$MODEL_PATH \
    cluster=single \
    apiserver.host=$host \
    apiserver.port=$port \
    > $RESULT_PATH/api_server.log 2>&1 &

wait_api_server $host $port

echo "start to send requests"
conda run -n dxz_dev --no-capture-output \
    python ${OUR_ROOT_PATH}/benchmark/benchmark.py \
    --num-requests=$NUM_REQUESTS \
    --model=${MODEL} \
    --model-path=${MODEL_PATH} \
    --host $host \
    --port $port \
    --result-path=$RESULT_PATH/result.json \
    --request-rate $REQUEST_RATES \
    --textcaps=1

clean_up