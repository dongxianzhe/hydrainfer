#!/bin/bash 
source ../common.sh

############################## params ##############################
# REQUEST_RATES="0.5 1 1.5 2 2.5 3 3.5 4 4.5 5"
# NUM_REQUESTS=30
REQUEST_RATES=5
NUM_REQUESTS=3
ttft_slo=8
tpot_slo=0.2
timeout=30
host="127.0.0.1"
port=$(find_free_port)
start_server_max_retry=5
find_free_gpus_max_retry=1000
only_text=0
start_server=1
start_benchmark=1
log_latency_breakdown=true
gpu_configs=(
    1
    # 2
    # 3
    # 4
    # 8
    # 16
    # 32
)
declare -A methods=(
    # ["ours"]="start_hydrainfer_server"
    # ["vllm"]="start_vllm_server"
    ["vllm-0-11-0"]="start_vllm_server"
    # ["vllm-0-10-2"]="start_vllm_server"
    # ["vllm-0-9-2"]="start_vllm_server"
    # ["vllm-0-8-5"]="start_vllm_server"
    # ["vllm-0-7-3"]="start_vllm_server"
    # ["vllm-0-6-6"]="start_vllm_server"
    # ["sglang"]="start_sglang_server"
    # ["sglang-0-5-3"]=start_sglang_server
    # ["sglang-0-4-10"]=start_sglang_server
    # ["sglang-0-4-3"]=start_sglang_server
    # ["sglang-0-3-6"]=start_sglang_server
    # ["sglang-0-2-15"]=start_sglang_server
    # ["lmdeploy"]="start_lmdeploy_server"
)
additional_server_configs=(
    ""
)
declare -A MODELS=(
    # ["llava-hf/llava-1.5-7b-hf"]="/models/llava-1.5-7b-hf"
    ["llava-hf/llava-v1.6-vicuna-7b-hf"]="/models/llava-v1.6-vicuna-7b-hf"
    # ["Qwen/Qwen2-VL-7B"]="/models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024"
    # ["deepseek-ai/deepseek-vl2-tiny"]="/models/deepseek-vl2-tiny"
    # ["OpenGVLab/InternVL2-26B"]="/models/OpenGVLab/InternVL2-26B"
)
trace_configs=(
    "--textcaps=1 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=0 --request-rate-method=poisson"
    # "--textcaps=0 --pope=1 --mme=0 --text_vqa=0 --vizwiz_vqa=0 --request-rate-method=poisson"
    # "--textcaps=0 --pope=0 --mme=1 --text_vqa=0 --vizwiz_vqa=0 --request-rate-method=poisson"
    # "--textcaps=0 --pope=0 --mme=0 --text_vqa=1 --vizwiz_vqa=0 --request-rate-method=poisson"
    # "--textcaps=0 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=1 --request-rate-method=poisson"
    # "--textcaps=1 --pope=1 --mme=1 --text_vqa=1 --vizwiz_vqa=1 --request-rate-method=poisson"
    # "--textcaps=1 --pope=1 --mme=1 --text_vqa=1 --vizwiz_vqa=1 --request-rate-method=azure_code"
    # "--textcaps=1 --pope=1 --mme=1 --text_vqa=1 --vizwiz_vqa=1 --request-rate-method=azure_conv"
    # "--textcaps=1 --pope=1 --mme=1 --text_vqa=1 --vizwiz_vqa=1 --request-rate-method=burstgpt"
    # "--textcaps=1 --pope=1 --mme=1 --text_vqa=1 --vizwiz_vqa=1 --request-rate-method=mooncake"
)
####################################################################

start_hydrainfer_server(){
    log_server_path="$RESULT_PATH/${timestamp}-log_server.jsonl"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    RAY_DEDUP_LOGS=0 \
        conda run -n hydrainfer --no-capture-output \
        python -m hydrainfer.entrypoint.entrypoint \
        model.path=$MODEL_PATH \
        apiserver.host=$host \
        apiserver.port=$port \
        ignore_eos=true \
        ttft_slo=$ttft_slo \
        tpot_slo=$tpot_slo \
        log_latency_breakdown=$log_latency_breakdown \
        cluster.log_server_path=${log_server_path} \
        > $api_server_log_path 2>&1 &
    # cluster=general cluster.n_enode=0 cluster.n_epnode=0 cluster.n_ednode=2 cluster.n_epdnode=0 cluster.n_pnode=6 cluster.n_pdnode=0 cluster.n_dnode=0 \
}

start_vllm_server(){
    extra=""

    if [[ "$method" == "vllm-0-7-3" && "$MODEL" == "llava-hf/llava-v1.6-vicuna-7b-hf" ]]; then
        extra="--chat-template=${OUR_ROOT_PATH}/hydrainfer/model/chat_template/template_llava.jinja"
    fi
    if [[ "$method" == "vllm-0-6-6" && "$MODEL" == "llava-hf/llava-v1.6-vicuna-7b-hf" ]]; then
        extra="--chat-template=${OUR_ROOT_PATH}/hydrainfer/model/chat_template/template_llava.jinja"
    fi

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    conda run -n ${method} --no-capture-output \
        vllm serve $MODEL_PATH \
        --host=$host \
        --port=$port \
        --tensor-parallel-size=${number_gpus_need} \
        --enforce-eager \
        $additional_server_config \
        $extra \
        > $api_server_log_path 2>&1 &
}

start_sglang_server(){
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    conda run -n ${method} --no-capture-output \
        python -m sglang.launch_server \
        --model-path=$MODEL_PATH \
        --host=$host \
        --port=$port \
        --disable-cuda-graph \
        --disable-cuda-graph-padding \
        --enable-multimodal \
        $additional_server_config \
        > $api_server_log_path 2>&1 &
}

start_lmdeploy_server(){
    # lmdeploy serve api_server -h
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    conda run -n ${method} --no-capture-output \
        lmdeploy serve api_server ${MODEL_PATH} \
        --eager-mode \
        --server-name=$host \
        --server-port=$port \
        $additional_server_config \
        > $api_server_log_path 2>&1 &
}

start_apiserver(){
    echo "starting api server"
    attempt=0
    while [ $attempt -lt $start_server_max_retry ]; do
        timestamp=$(date +"%Y%m%d_%H%M%S")
        api_server_log_path="$RESULT_PATH/${log_prefix}-${timestamp}-api_server.log"
        ln -sf $api_server_log_path latest_api_server.log

        $server_start_method
        apiserver_pid=$!
        if wait_api_server $host $port $apiserver_pid; then
            echo "server start success."
            break
        else
            echo "server start failed. Retrying..."
            clean_up
            attempt=$((attempt + 1))
        fi
    done
}

send_requests(){
    echo "start to send requests"
    safe_trace=$(echo "$trace" | sed 's/[ \/:*?"<>|&!(){}]/_/g')
    conda run -n hydrainfer --no-capture-output \
        python ${OUR_ROOT_PATH}/benchmark/benchmark.py \
        --model=${MODEL} \
        --model-path=${MODEL_PATH} \
        --host $host \
        --port $port \
        --num-requests=$NUM_REQUESTS \
        --request-rate-num-requests-scale=$number_gpus_need \
        --request-rate $REQUEST_RATES \
        --timeout=${timeout} \
        --only_text=${only_text} \
        $trace \
        --show-result=4 \
        --backend=${method} \
        --method-name=${method} \
        --result-path=$RESULT_PATH/${log_prefix}-${safe_trace}-result.json
}

evaluation(){
    if [[ "$start_server" == "1" ]]; then
        start_apiserver
    fi

    if [[ "$start_benchmark" == "1" ]]; then
        for trace in "${trace_configs[@]}"; do
            if ! wait_api_server "$host" "$port" "$apiserver_pid" && [[ "$start_server" == "1" ]]; then
                echo "server failed after benchmark. try to restart server."
                start_apiserver
            fi
            send_requests
        done
    else
        echo "server is running..."
        sleep 7200
    fi

    if [[ "$log_latency_breakdown" == "true" || "$method" != "ours" ]]; then
        conda run -n hydrainfer --no-capture-output \
            python latency_breakdown.py \
            --path=${log_server_path} \
            > $RESULT_PATH/${log_prefix}-${timestamp}-latency_breakdown.log 2>&1
    else
        echo "Latency breakdown logging is disabled."
        fi

    clean_up
    sleep 10
}

for method in "${!methods[@]}"; do
    server_start_method="${methods[$method]}"
    RESULT_PATH=$(echo "$SCRIPT_PATH/result/${method}/$(date +%Y%m%d_%H%M%S)")
    mkdir -p $RESULT_PATH
    for number_gpus_need in ${gpu_configs[@]}; do
        for MODEL in "${!MODELS[@]}"; do
            MODEL_PATH="${MODELS[$MODEL]}"
            for additional_server_config in "${additional_server_configs[@]}"; do
                attempt=0
                while [ $attempt -lt $find_free_gpus_max_retry ]; do
                    read -a free_gpus <<< "$(get_free_gpus)"
                    selected_gpus=("${free_gpus[@]:0:$number_gpus_need}")
                    gpu_list=$(IFS=,; echo "${selected_gpus[*]}")
                    CUDA_VISIBLE_DEVICES="$gpu_list"
                    echo "=============================================="
                    echo "Evaluating methods: $method"
                    echo "model: $MODEL"
                    echo "Path: $MODEL_PATH"
                    echo "additional_server_config: $additional_server_config"
                    echo "free gpus ${free_gpus[@]}"
                    echo "number_gpus_need ${number_gpus_need}"
                    echo "set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
                    echo "=============================================="
                    safe_gpus_info="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
                    safe_method=$(echo "$method" | sed 's/[\/:*?"<>|&!(){}]/_/g')
                    safe_additional_server_config=$(echo "$additional_server_config" | sed 's/[\/:*?"<>|&!(){}]/_/g')
                    safe_model=$(echo "$MODEL" | sed 's/[\/:*?"<>|&!(){}]/_/g')
                    log_prefix="${safe_method}-scale${safe_gpus_info}-${safe_additional_server_config}-${safe_model}"

                    if [[ ${#free_gpus[@]} -ge "$number_gpus_need" || "$start_server" == "0" ]]; then
                        evaluation
                        break
                    else
                        echo "not enough available gpus. retrying..."
                        clean_up
                        sleep 60
                        attempt=$((attempt + 1))
                    fi
                done
            done
        done
    done
done