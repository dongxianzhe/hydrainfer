#!/bin/bash 

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export RAY_DEDUP_LOGS=0

MODEL_PATH=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf
APISERVER_PORT=8888

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
RESULT_DIR="$SCRIPT_PATH/result"

clean_up() {
    echo "Cleaning up..."
    pgrep -f "dxz.entrypoint.entrypoint" >/dev/null && pgrep -f "dxz.entrypoint.entrypoint" | xargs kill
}

trap clean_up EXIT

mkdir -p $RESULT_DIR

cd ../../benchmark

node_configs=(
    #e  ep ed epd p pd d
    "0  0  0  1  0  0  0"
    "0  0  1  0  1  0  0"  
    "0  1  0  0  0  0  1"  
    "1  0  0  0  1  0  1"  
)

for config in "${node_configs[@]}"; do
    read -r n_enode n_epnode n_ednode n_epdnode n_pnode n_pdnode n_dnode <<< "$config"
    config_name="e$n_enode-ep$n_epnode-ed$n_ednode-epd$n_epdnode-p$n_pnode-pd$n_pdnode-d$n_dnode"

    echo evaluating $config_name
    RAY_DEDUP_LOGS=0 python -m dxz.entrypoint.entrypoint \
        model.path=${MODEL_PATH} \
        cluster=general \
        cluster.n_enode=$n_enode \
        cluster.n_epnode=$n_epnode \
        cluster.n_ednode=$n_ednode \
        cluster.n_epdnode=$n_epdnode \
        cluster.n_pnode=$n_pnode \
        cluster.n_pdnode=$n_pdnode \
        cluster.n_dnode=$n_dnode \
        apiserver.port=$APISERVER_PORT \
        > $RESULT_DIR/${config_name}_api_server.log 2>&1 &

    retry=0
    while ! nc -z 127.0.0.1 $APISERVER_PORT; do
        echo "Waiting for apiserver to start..."
        retry=$((retry + 1))
        if [ $retry -gt 50 ]; then
            echo "apiserver failed to start after 50 attempts. Exiting."
            exit 1
        fi
        sleep 5
    done

    echo "apiserver is running on port $APISERVER_PORT"
    echo "Start benchmarking"

    conda run -n dxz_dev --no-capture-output \
        python benchmark.py \
            --num-requests=100 \
            --backend=dxz \
            --inference-mode=online \
            --model-path=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf \
            --log-requests \
            --summary=0 \
            --msummary=0 \
            --chat=0 \
            --mchat=1 \
            --log-output \
            --request-rate 12 \
            > $RESULT_DIR/${config_name}_result.log

    clean_up
    sleep 3
done
