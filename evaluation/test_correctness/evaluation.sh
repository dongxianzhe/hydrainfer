#!/bin/bash 

source ../common.sh

export CUDA_VISIBLE_DEVICES=1
host="127.0.0.1"
port="8888"

echo "start test inference"
RAY_DEDUP_LOGS=0 \
    conda run -n dxz_dev --no-capture-output \
    python -m dxz.entrypoint.entrypoint \
    model.path=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf \
    cluster=single \
    apiserver.host=$host \
    apiserver.port=$port \
    > $RESULT_DIR/api_server.log 2>&1 &

wait_api_server $host $port

echo "Start benchmarking"
conda run -n dxz_dev --no-capture-output \
    python ${OUR_ROOT_DIR}/benchmark/benchmark.py \
    --num-requests=16 --backend=dxz --inference-mode=online --model-path=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf --mtest=1 --port=8888 --log-request --log-output

clean_up