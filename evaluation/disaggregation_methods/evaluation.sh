#!/bin/bash 

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
    # ED+P
    # "0  0  1  0  7  0  0"
    # "0  0  2  0  6  0  0"
    # "0  0  3  0  5  0  0"
    # "0  0  4  0  4  0  0"
    # "0  0  5  0  3  0  0"
    # "0  0  6  0  2  0  0"
    # "0  0  7  0  1  0  0"

    # EP+D
    # "0  1  0  0  0  0  7"  
    # "0  2  0  0  0  0  6"  
    # "0  3  0  0  0  0  5"  
    # "0  4  0  0  0  0  4"  
    # "0  5  0  0  0  0  3"  
    # "0  6  0  0  0  0  2"  
    # "0  7  0  0  0  0  1"  

    # ED+P+D
    # "0  0  6  0  1  0  1"
    # "0  0  5  0  2  0  1"
    # "0  0  5  0  1  0  2"
    # "0  0  4  0  3  0  1"
    # "0  0  4  0  2  0  2"
    # "0  0  4  0  1  0  3"
    # "0  0  3  0  4  0  1"
    # "0  0  3  0  3  0  2"
    # "0  0  3  0  2  0  3"
    # "0  0  3  0  1  0  4"
    # "0  0  2  0  5  0  1"
    # "0  0  2  0  4  0  2"
    # "0  0  2  0  3  0  3"
    # "0  0  2  0  2  0  4"
    # "0  0  2  0  1  0  5"
    # "0  0  1  0  6  0  1"
    # "0  0  1  0  5  0  2"
    # "0  0  1  0  4  0  3"
    # "0  0  1  0  3  0  4"
    # "0  0  1  0  2  0  5"
    # "0  0  1  0  1  0  6"

    # EPD scaling
    # "0  0  0  1  0  0  0"
    # "0  0  0  2  0  0  0"
    # "0  0  0  3  0  0  0"
    # "0  0  0  4  0  0  0"
    # "0  0  0  5  0  0  0"
    # "0  0  0  6  0  0  0"
    # "0  0  0  7  0  0  0"
    # "0  0  0  8  0  0  0"

    # EP+P+D ED+P+D 8 
    # "0 0 1 0 1 0 6"
    # "0 0 2 0 1 0 5"
    # "0 0 3 0 1 0 4"
    # "0 0 4 0 1 0 3"
    # "0 0 5 0 1 0 2"
    # "0 0 6 0 1 0 1"
    # "0 0 7 0 1 0 0"
    # "0 1 0 0 0 0 7"
    # "0 0 1 0 2 0 5"
    # "0 0 2 0 2 0 4"
    # "0 0 3 0 2 0 3"
    # "0 0 4 0 2 0 2"
    # "0 0 5 0 2 0 1"
    # "0 0 6 0 2 0 0"
    # "0 1 0 0 1 0 6"
    # "0 2 0 0 0 0 6"
    # "0 0 1 0 3 0 4"
    # "0 0 2 0 3 0 3"
    # "0 0 3 0 3 0 2"
    # "0 0 4 0 3 0 1"
    # "0 0 5 0 3 0 0"
    # "0 1 0 0 2 0 5"
    # "0 2 0 0 1 0 5"
    # "0 3 0 0 0 0 5"
    # "0 0 1 0 4 0 3"
    # "0 0 2 0 4 0 2"
    # "0 0 3 0 4 0 1"
    # "0 0 4 0 4 0 0"
    # "0 1 0 0 3 0 4"
    # "0 2 0 0 2 0 4"
    # "0 3 0 0 1 0 4"
    # "0 4 0 0 0 0 4"
    # "0 0 1 0 5 0 2"
    # "0 0 2 0 5 0 1"
    # "0 0 3 0 5 0 0"
    # "0 1 0 0 4 0 3"
    # "0 2 0 0 3 0 3"
    # "0 3 0 0 2 0 3"
    # "0 4 0 0 1 0 3"
    # "0 5 0 0 0 0 3"
    # "0 0 1 0 6 0 1"
    # "0 0 2 0 6 0 0"
    # "0 1 0 0 5 0 2"
    # "0 2 0 0 4 0 2"
    # "0 3 0 0 3 0 2"
    # "0 4 0 0 2 0 2"
    # "0 5 0 0 1 0 2"
    # "0 6 0 0 0 0 2"
    # "0 0 1 0 7 0 0"
    # "0 1 0 0 6 0 1"
    # "0 2 0 0 5 0 1"
    # "0 3 0 0 4 0 1"
    # "0 4 0 0 3 0 1"
    # "0 5 0 0 2 0 1"
    # "0 6 0 0 1 0 1"
    # "0 7 0 0 0 0 1"

    # EP+P+D ED+P+D 7 
    # "0 0 1 0 1 0 5"
    # "0 0 2 0 1 0 4"
    # "0 0 3 0 1 0 3"
    # "0 0 4 0 1 0 2"
    # "0 0 5 0 1 0 1"
    # "0 0 6 0 1 0 0"
    # "0 1 0 0 0 0 6"
    # "0 0 1 0 2 0 4"
    # "0 0 2 0 2 0 3"
    # "0 0 3 0 2 0 2"
    # "0 0 4 0 2 0 1"
    # "0 0 5 0 2 0 0"
    # "0 1 0 0 1 0 5"
    # "0 2 0 0 0 0 5"
    # "0 0 1 0 3 0 3"
    # "0 0 2 0 3 0 2"
    # "0 0 3 0 3 0 1"
    # "0 0 4 0 3 0 0"
    # "0 1 0 0 2 0 4"
    # "0 2 0 0 1 0 4"
    # "0 3 0 0 0 0 4"
    # "0 0 1 0 4 0 2"
    # "0 0 2 0 4 0 1"
    # "0 0 3 0 4 0 0"
    # "0 1 0 0 3 0 3"
    # "0 2 0 0 2 0 3"
    # "0 3 0 0 1 0 3"
    # "0 4 0 0 0 0 3"
    # "0 0 1 0 5 0 1"
    # "0 0 2 0 5 0 0"
    # "0 1 0 0 4 0 2"
    # "0 2 0 0 3 0 2"
    # "0 3 0 0 2 0 2"
    # "0 4 0 0 1 0 2"
    # "0 5 0 0 0 0 2"
    # "0 0 1 0 6 0 0"
    # "0 1 0 0 5 0 1"
    # "0 2 0 0 4 0 1"
    # "0 3 0 0 3 0 1"
    # "0 4 0 0 2 0 1"
    # "0 5 0 0 1 0 1"
    # "0 6 0 0 0 0 1"

    #e  ep ed epd p pd d
    #E + P + D 7
    # "1 0 0 0 1 0 5"
    # "2 0 0 0 1 0 4"
    # "3 0 0 0 1 0 3"
    # "4 0 0 0 1 0 2"
    # "5 0 0 0 1 0 1"
    # "1 0 0 0 2 0 4"
    # "2 0 0 0 2 0 3"
    # "3 0 0 0 2 0 2"
    # "4 0 0 0 2 0 1"
    # "1 0 0 0 3 0 3"
    # "2 0 0 0 3 0 2"
    # "3 0 0 0 3 0 1"
    # "1 0 0 0 4 0 2"
    # "2 0 0 0 4 0 1"
    # "1 0 0 0 5 0 1"

    # EP + ED + P + D 8 node 56
    # "0 0 1 0 1 0 6"
    # "0 0 2 0 1 0 5"
    # "0 0 3 0 1 0 4"
    # "0 0 4 0 1 0 3"
    # "0 0 5 0 1 0 2"
    # "0 0 6 0 1 0 1"
    # "0 0 7 0 1 0 0"
    # "0 1 0 0 0 0 7"
    # "0 0 1 0 2 0 5"
    # "0 0 2 0 2 0 4"
    # "0 0 3 0 2 0 3"
    # "0 0 4 0 2 0 2"
    # "0 0 5 0 2 0 1"
    # "0 0 6 0 2 0 0"
    # "0 1 0 0 1 0 6"
    # "0 2 0 0 0 0 6"
    # "0 0 1 0 3 0 4"
    # "0 0 2 0 3 0 3"
    # "0 0 3 0 3 0 2"
    # "0 0 4 0 3 0 1"
    # "0 0 5 0 3 0 0"
    # "0 1 0 0 2 0 5"
    # "0 2 0 0 1 0 5"
    # "0 3 0 0 0 0 5"
    # "0 0 1 0 4 0 3"
    # "0 0 2 0 4 0 2"
    # "0 0 3 0 4 0 1"
    # "0 0 4 0 4 0 0"
    # "0 1 0 0 3 0 4"
    # "0 2 0 0 2 0 4"
    # "0 3 0 0 1 0 4"
    # "0 4 0 0 0 0 4"
    # "0 0 1 0 5 0 2"
    # "0 0 2 0 5 0 1"
    # "0 0 3 0 5 0 0"
    # "0 1 0 0 4 0 3"
    # "0 2 0 0 3 0 3"
    # "0 3 0 0 2 0 3"
    # "0 4 0 0 1 0 3"
    # "0 5 0 0 0 0 3"
    # "0 0 1 0 6 0 1"
    # "0 0 2 0 6 0 0"
    # "0 1 0 0 5 0 2"
    # "0 2 0 0 4 0 2"
    # "0 3 0 0 3 0 2"
    # "0 4 0 0 2 0 2"
    # "0 5 0 0 1 0 2"
    # "0 6 0 0 0 0 2"
    # "0 0 1 0 7 0 0"
    # "0 1 0 0 6 0 1"
    # "0 2 0 0 5 0 1"
    # "0 3 0 0 4 0 1"
    # "0 4 0 0 3 0 1"
    # "0 5 0 0 2 0 1"
    # "0 6 0 0 1 0 1"
    # "0 7 0 0 0 0 1"

    # E + P + D 8 node 21
    # "1 0 0 0 1 0 6"
    # "2 0 0 0 1 0 5"
    # "3 0 0 0 1 0 4"
    # "4 0 0 0 1 0 3"
    # "5 0 0 0 1 0 2"
    # "6 0 0 0 1 0 1"
    # "1 0 0 0 2 0 5"
    # "2 0 0 0 2 0 4"
    # "3 0 0 0 2 0 3"
    # "4 0 0 0 2 0 2"
    # "5 0 0 0 2 0 1"
    # "1 0 0 0 3 0 4"
    # "2 0 0 0 3 0 3"
    # "3 0 0 0 3 0 2"
    # "4 0 0 0 3 0 1"
    # "1 0 0 0 4 0 3"
    # "2 0 0 0 4 0 2"
    # "3 0 0 0 4 0 1"
    # "1 0 0 0 5 0 2"
    # "2 0 0 0 5 0 1"
    # "1 0 0 0 6 0 1"

    "0 0 5 0 3 0 0"
    "1 0 0 0 3 0 4"
    "0 4 0 0 0 0 4"
    "0 0 0 8 0 0 0"
)

request_rate="8"
# request_rate="1 2 3 4 5 6 7 8 9 10"

scenarios=(
    # "--mchat=1"
    # "--chat=1"
    # "--msummary=1"
    # "--summary=1"
    # "--mchat=1 --chat=1 --msummary=1 --summary=1"
    # "--mme=1"
    "--textcaps=1"
)


for config in "${node_configs[@]}"; do
    read -r n_enode n_epnode n_ednode n_epdnode n_pnode n_pdnode n_dnode <<< "$config"
    config_name="e$n_enode-ep$n_epnode-ed$n_ednode-epd$n_epdnode-p$n_pnode-pd$n_pdnode-d$n_dnode"

    sum_node=$((n_enode + n_epnode + n_ednode + n_epdnode + n_pnode + n_pdnode + n_dnode))

    echo evaluating $config_name
    conda run -n dxz_dev --no-capture-output \
        python -m dxz.entrypoint.entrypoint \
            model.path=${MODEL_PATH} \
            cluster=general \
            cluster.n_enode=$n_enode \
            cluster.n_epnode=$n_epnode \
            cluster.n_ednode=$n_ednode \
            cluster.n_epdnode=$n_epdnode \
            cluster.n_pnode=$n_pnode \
            cluster.n_pdnode=$n_pdnode \
            cluster.n_dnode=$n_dnode \
            cluster.ednode.log_step_latency=True \
            apiserver.port=$APISERVER_PORT \
            cluster.ednode.executor.multi_streams_forward=False \
            cluster.ednode.executor.multi_threads_forward=False \
            cluster.ednode.batch_scheduler.debug=True \
            ignore_eos=False \
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

    scaled_request_rate=""
    for rate in $request_rate; do
        scaled_rate=$((rate * sum_node))
        
        if [ -z "$scaled_request_rate" ]; then
            scaled_request_rate="$scaled_rate"
        else
            scaled_request_rate="$scaled_request_rate $scaled_rate"
        fi
    done

    for scenario in "${scenarios[@]}"; do
        echo "Running scenario: $scenario request_rate $scaled_request_rate"

        conda run -n dxz_dev --no-capture-output \
            python benchmark.py \
                --num-requests=512 \
                --backend=dxz \
                --inference-mode=online \
                --model-path=/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf \
                --log-requests \
                $scenario \
                --log-output \
                --request-rate $scaled_request_rate\
                > $RESULT_DIR/${config_name}_${scenario// /_}_result.log
    done

    clean_up
    sleep 3
done

conda run -n dxz_dev --no-capture-output \
    python $SCRIPT_PATH/get_metric.py --folder $RESULT_DIR > $RESULT_DIR/metric.log