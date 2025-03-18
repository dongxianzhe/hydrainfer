#!/bin/bash 

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")

RESULT_DIR="$SCRIPT_PATH/result"

export CUDA_VISIBLE_DEVICES=0

clean_up() {
    echo "Cleaning up..."
    pgrep -f "vllm serve" >/dev/null && pgrep -f "vllm serve" | xargs kill
    pgrep -f "dxz.entrypoint.entrypoint" >/dev/null && pgrep -f "dxz.entrypoint.entrypoint" | xargs kill
}

trap clean_up EXIT

mkdir -p $RESULT_DIR

cd ../../benchmark
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

evaluate_sarathi() {
    echo "Evaluating sarathi"

    RAY_DEDUP_LOGS=0 \
    conda run -n dxz_dev --no-capture-output \
        python -m dxz.entrypoint.entrypoint \
        model.path=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf cluster=single \
        cluster.epdnode.kv_cache.n_blocks=2048 \
        cluster.epdnode.batch_scheduler_profiler.profile_batch_config=False \
        cluster.epdnode.batch_scheduler.max_running_requests=24 \
        cluster.epdnode.batch_scheduler.max_batch_fill_tokens=2048 \
        cluster.epdnode.batch_scheduler.chunked_prefill=True \
        cluster.epdnode.request_processor.disaggregate_embed_prefill=False \
        > $RESULT_DIR/sarathi_api_server.log 2>&1 &

    retry=0
    while ! nc -z 127.0.0.1 8888; do
        echo "Waiting for sarathi to start..."
        retry=$((retry + 1))
        if [ $retry -gt 50 ]; then
            echo "sarathi failed to start after 50 attempts. Exiting."
            exit 1
        fi
        sleep 5
    done

    echo "sarathi is running on port 8888"
    echo "Start benchmarking"

    conda run -n dxz_dev --no-capture-output python benchmark.py --num-requests=256 --backend=dxz --inference-mode=online --model-path=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf --stall=1 --log-token-times --port=8888 > $RESULT_DIR/sarathi_result.log

    echo "Finished evaluating sarathi"
    clean_up
}

evaluate_ours() {
    echo "Evaluating ours"

    RAY_DEDUP_LOGS=0 \
    conda run -n dxz_dev --no-capture-output \
        python -m dxz.entrypoint.entrypoint \
        model.path=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf cluster=single \
        cluster.epdnode.batch_scheduler.debug=True \
        cluster.epdnode.kv_cache.n_blocks=2048 \
        cluster.epdnode.batch_scheduler_profiler.profile_batch_config=True \
        cluster.epdnode.batch_scheduler_profiler.profile_image_budgets=True \
        cluster.epdnode.batch_scheduler_profiler.profile_token_budgets=True \
        cluster.epdnode.batch_scheduler.max_running_requests=24 \
        cluster.epdnode.batch_scheduler.chunked_prefill=True \
        cluster.epdnode.request_processor.disaggregate_embed_prefill=True \
        > $RESULT_DIR/dxz_api_server.log 2>&1 &

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

    conda run -n dxz_dev --no-capture-output python benchmark.py --num-requests=256 --backend=dxz --inference-mode=online --model-path=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf --stall=1 --log-token-times --port=8888 > $RESULT_DIR/dxz_result.log

    echo "Finished evaluating dxz"
    clean_up
}

evaluate_vllm
sleep 3
evaluate_sarathi
sleep 3
evaluate_ours