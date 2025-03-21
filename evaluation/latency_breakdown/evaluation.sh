#!/bin/bash 

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")

RESULT_DIR="$SCRIPT_PATH/result"

export CUDA_VISIBLE_DEVICES=1,2,3

clean_up() {
    echo "Cleaning up..."
    pgrep -f "dxz.entrypoint.entrypoint" >/dev/null && pgrep -f "dxz.entrypoint.entrypoint" | xargs kill
}

trap clean_up EXIT
cd ../../benchmark

mkdir -p $RESULT_DIR

RAY_DEDUP_LOGS=0 \
conda run -n dxz_dev --no-capture-output \
    python -m dxz.entrypoint.entrypoint \
    model.path=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf cluster=hybrid \
    log_latency_breakdown=True \
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

echo "apiserver is running on port 8888"
echo "Start benchmarking"

conda run -n dxz_dev --no-capture-output python benchmark.py --num-requests=2 --backend=dxz --inference-mode=online --model-path=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf --textcaps=1 --port=8888 --log-request --log-output > /dev/null 2>&1

sleep 3
clean_up
sleep 3

cd $SCRIPT_PATH
echo "Start latency breakdown analysis"
conda run -n dxz_dev --no-capture-output python $SCRIPT_PATH/latency_breakdown_analysis.py > $RESULT_DIR/latency_breakdown.log 2>&1 &

echo "Finished evaluating dxz"
clean_up