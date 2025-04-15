#!/bin/bash 

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")

RESULT_DIR="$SCRIPT_PATH/result"

export CUDA_VISIBLE_DEVICES=1,2,3

clean_up() {
    echo "Cleaning up..."
    pgrep -f "vllm serve" >/dev/null && pgrep -f "vllm serve" | xargs kill
    pgrep -f "dxz.entrypoint.entrypoint" >/dev/null && pgrep -f "dxz.entrypoint.entrypoint" | xargs kill
}

trap clean_up EXIT
cd ../../benchmark

mkdir -p $RESULT_DIR

evaluate_vllm() {
    echo "Evaluating vllm"
    conda run -n vllm --no-capture-output \
        vllm serve /mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf \
        --port=8888 \
        --chat-template=/export/home/dongxianzhe1/projects/vllm/examples/template_llava.jinja \
        --enforce-eager > $RESULT_DIR/vllm_api_server.log 2>&1 &
    retry=0
    while ! nc -z 127.0.0.1 8888; do
        echo "Waiting for vllm to start..."
        retry=$((retry + 1))
        if [ $retry -gt 50 ]; then
            echo "vllm failed to start after 50 attempts. Exiting."
            exit 1
        fi
        sleep 5
    done
    echo "vllm is running on port 8888"
    echo "Start benchmarking"

    conda run -n dxz_dev --no-capture-output python benchmark.py --num-requests=256 --backend=vllm --inference-mode=online --model-path=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf --stall=1 --log-token-times --port=8888 > $RESULT_DIR/vllm_result.log

    echo "Finished evaluating vllm"
    clean_up
}

evaluate_ours() {
    echo "Evaluating ours"

    RAY_DEDUP_LOGS=0 \
    conda run -n dxz_dev --no-capture-output \
        python -m dxz.entrypoint.entrypoint \
        model.path=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf cluster=hybrid \
        log_latency_breakdown=False \
        > $RESULT_DIR/api_server.log 2>&1 &

    retry=0
    while ! nc -z 127.0.0.1 8888; do
        echo "Waiting for dxz to start..."
        retry=$((retry + 1))
        if [ $retry -gt 50 ]; then
            echo "dxz failed to start after 50 attempts. Exiting."
            exit 1
        fi
        sleep 5
    done

    echo "dxz is running on port 8888"
    echo "Start benchmarking"

    conda run -n dxz_dev --no-capture-output python benchmark.py --num-requests=16 --backend=dxz --inference-mode=online --model-path=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf --mtest=1 --port=8888 --log-request --log-output

    clean_up
}

evaluate_ours