#!/bin/bash 

source ../common.sh

export CUDA_VISIBLE_DEVICES=1
MODEL="llava-hf/llava-1.5-7b-hf"
MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf"
CHAT_TEMPLATE_PATH=$OUR_ROOT_PATH/dxz/chat_template/template_llava.jinja

MODEL="llava-hf/llava-v1.6-vicuna-7b-hf"
MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/llava-v1.6-vicuna-7b-hf"
CHAT_TEMPLATE_PATH=$OUR_ROOT_PATH/dxz/chat_template/template_llava.jinja

MODEL="Qwen/Qwen2-VL-7B"
MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024"
CHAT_TEMPLATE_PATH=$OUR_ROOT_PATH/dxz/chat_template/template_qwen_vl_chat.jinja

REQUEST_RATES=1
NUM_REQUESTS=3
host="127.0.0.1"
port="8891"

echo "starting api server"
RAY_DEDUP_LOGS=0 \
    conda run -n dxz_dev --no-capture-output \
    python -m dxz.entrypoint.entrypoint \
    model=qwen2vl7b \
    model.name=$MODEL \
    model.path=$MODEL_PATH \
    cluster.epdnode.executor.use_flash_infer=false \
    cluster.epdnode.executor.multi_streams_forward=false \
    cluster.epdnode.executor.multi_threads_forward=false \
    cluster=single \
    cluster.epdnode.kv_cache.n_blocks=2048 \
    apiserver.chat_template=$CHAT_TEMPLATE_PATH \
    apiserver.host=$host \
    apiserver.port=$port \
    ignore_eos=true \
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
    --backend=ours \
    --textcaps=1

clean_up