#!/bin/bash 

source ../common.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL="llava-hf/llava-1.5-7b-hf"
MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf"
CHAT_TEMPLATE_PATH=$OUR_ROOT_PATH/dxz/chat_template/template_llava.jinja

# MODEL="llava-hf/llava-v1.6-vicuna-7b-hf"
# MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/llava-v1.6-vicuna-7b-hf"
# CHAT_TEMPLATE_PATH=$OUR_ROOT_PATH/dxz/chat_template/template_llava.jinja

# MODEL="liuhaotian/llava-v1.5-7b"
# MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/liuhaotian/llava-v1.5-7b"
# CHAT_TEMPLATE_PATH=$OUR_ROOT_PATH/dxz/chat_template/template_llava.jinja

# MODEL="liuhaotian/llava-v1.6-vicuna-7b"
# MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/liuhaotian/llava-v1.6-vicuna-7b"
# CHAT_TEMPLATE_PATH=$OUR_ROOT_PATH/dxz/chat_template/template_llava.jinja

# MODEL="Qwen/Qwen2-VL-7B"
# MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024"
# CHAT_TEMPLATE_PATH=$OUR_ROOT_PATH/dxz/chat_template/template_qwen_vl_chat.jinja

# MODEL="Qwen/Qwen2-7B-Instruct"
# MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/Qwen2-7B-Instruct"
# CHAT_TEMPLATE_PATH=$OUR_ROOT_PATH/dxz/chat_template/template_qwen_vl_chat.jinja

# MODEL="deepseek-ai/deepseek-vl2-tiny"
# MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/deepseek-vl2-tiny"
# CHAT_TEMPLATE_PATH=$OUR_ROOT_PATH/dxz/chat_template/template_deepseek_vl2.jinja

# cp /mnt/cfs/9n-das-admin/llm_models/lmms-lab/llama3-llava-next-8b/preprocessor_config.json /mnt/cfs/9n-das-admin/llm_models/liuhaotian/llava-v1.5-7b/preprocessor_config.json

LOG_LATENCY_BREAKDOWN=false
REQUEST_RATES="1 2 3 4 5 6 7 8 9 10 11 12"
NUM_REQUESTS=10
host="127.0.0.1"
port="8899"
disaggregation_methods_file="disaggregation_methods.json"

scenarios=(
    "--textcaps=0 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=1"
    "--textcaps=1 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=0"
    "--textcaps=0 --pope=1 --mme=0 --text_vqa=0 --vizwiz_vqa=0"
    "--textcaps=0 --pope=0 --mme=1 --text_vqa=0 --vizwiz_vqa=0"
    "--textcaps=0 --pope=0 --mme=0 --text_vqa=1 --vizwiz_vqa=0"
)

methods=(
    # "lmdeploy"
    # "lightllm"
    # "tgi"
    # "sglang"
    # "sglang3"
    # "sglang2"
    # "llama"
    # "vllm"
    # "vllm-v0"
    # "vllm6"
    # "vllm7"
    # "vllm8"
    # "vllm-tp8"
    # "ours-abligation-multistream"
    # "ours-abligation-batchpolicy"
    "ours"
)

echo "methods ${methods}"

sendrequest (){
    conda run -n dxz_dev --no-capture-output \
        python $OUR_ROOT_PATH/benchmark/benchmark.py \
        --host=$host \
        --port=$port \
        --num-requests=$NUM_REQUESTS \
        --request-rate-num-requests-scale=$num_gpu \
        --model=$MODEL \
        --model-path=$MODEL_PATH \
        --method-name=${method} \
        --request-rate ${REQUEST_RATES} \
        --backend=$method \
        $scenario \
        --result-path="$RESULT_PATH/${method}_${scenario// /_}_results.json"
}

benchmark() {
    if ! wait_api_server $host $port; then
        return 1
    fi
    echo "Start benchmarking ${method}"
    for scenario in "${scenarios[@]}"; do
        echo "Running scenario: $scenario"
        sendrequest
    done
    return 0
}

benchmark_one_scenario(){
    local scenario=$1
    if ! wait_api_server $host $port; then
        return 1
    fi
    echo "Start benchmarking ${method}"
    echo "Running scenario: $scenario"
    sendrequest
    return 0
}

start_our_server() {
    RAY_DEDUP_LOGS=0 \
    conda run -n dxz_dev --no-capture-output \
        python -m dxz.entrypoint.entrypoint \
        model.name=$MODEL \
        model.path=${MODEL_PATH} \
        apiserver.host=$host \
        apiserver.port=$port \
        zmq.port=40838 \
        apiserver.chat_template=$CHAT_TEMPLATE_PATH \
        nccl_communicator.host=$host \
        nccl_communicator.port=$((port + 1)) \
        ignore_eos=True \
        cluster=general \
        cluster.n_enode=$n_enode \
        cluster.n_epnode=$n_epnode \
        cluster.n_ednode=$n_ednode \
        cluster.n_epdnode=$n_epdnode \
        cluster.n_pnode=$n_pnode \
        cluster.n_pdnode=$n_pdnode \
        cluster.n_dnode=$n_dnode \
        cluster.debug=false \
        cluster.enode.image_cache.n_blocks=7000 \
        cluster.epnode.image_cache.n_blocks=250 \
        cluster.ednode.image_cache.n_blocks=250 \
        cluster.epdnode.image_cache.n_blocks=250 \
        cluster.pnode.image_cache.n_blocks=250 \
        cluster.pdnode.image_cache.n_blocks=250 \
        cluster.dnode.image_cache.n_blocks=0 \
        cluster.enode.kv_cache.n_blocks=0 \
        cluster.epnode.kv_cache.n_blocks=5000 \
        cluster.ednode.kv_cache.n_blocks=5000 \
        cluster.epdnode.kv_cache.n_blocks=5000 \
        cluster.pnode.kv_cache.n_blocks=6500 \
        cluster.pdnode.kv_cache.n_blocks=6500 \
        cluster.dnode.kv_cache.n_blocks=6500 \
        cluster.ednode.executor.multi_streams_forward=${multi_streams_forward} \
        cluster.ednode.executor.multi_threads_forward=${multi_threads_forward} \
        cluster.epnode.executor.multi_streams_forward=${multi_streams_forward} \
        cluster.epnode.executor.multi_threads_forward=${multi_threads_forward} \
        cluster.enode.batch_scheduler.max_running_requests=1200 \
        cluster.epnode.batch_scheduler.max_running_requests=24 \
        cluster.ednode.batch_scheduler.max_running_requests=24 \
        cluster.epdnode.batch_scheduler.max_running_requests=24 \
        cluster.pnode.batch_scheduler.max_running_requests=24 \
        cluster.pdnode.batch_scheduler.max_running_requests=24 \
        cluster.dnode.batch_scheduler.max_running_requests=24 \
        cluster.enode.executor.use_flash_infer=false \
        cluster.pnode.executor.use_flash_infer=false \
        cluster.dnode.executor.use_flash_infer=false \
        cluster.epnode.executor.use_flash_infer=false \
        cluster.ednode.executor.use_flash_infer=false \
        cluster.pdnode.executor.use_flash_infer=false \
        cluster.epdnode.executor.use_flash_infer=false \
        cluster.enode.log_latency_breakdown=$LOG_LATENCY_BREAKDOWN \
        cluster.pnode.log_latency_breakdown=$LOG_LATENCY_BREAKDOWN \
        cluster.dnode.log_latency_breakdown=$LOG_LATENCY_BREAKDOWN \
        cluster.epnode.log_latency_breakdown=$LOG_LATENCY_BREAKDOWN \
        cluster.ednode.log_latency_breakdown=$LOG_LATENCY_BREAKDOWN \
        cluster.pdnode.log_latency_breakdown=$LOG_LATENCY_BREAKDOWN \
        cluster.epdnode.log_latency_breakdown=$LOG_LATENCY_BREAKDOWN \
        > $RESULT_PATH/${method}_api_server.log 2>&1 &
}

start_batch_policy_abligation_study_server() {
    num_gpu=8
    RAY_DEDUP_LOGS=0 \
    conda run -n dxz_dev --no-capture-output \
        python -m dxz.entrypoint.entrypoint \
        model.name=$MODEL \
        model.path=${MODEL_PATH} \
        apiserver.host=$host \
        apiserver.port=$port \
        apiserver.chat_template=$CHAT_TEMPLATE_PATH \
        nccl_communicator.host=$host \
        nccl_communicator.port=$((port + 1)) \
        zmq.port=$((port + 2)) \
        ignore_eos=True \
        cluster=general \
        cluster.n_enode=0 \
        cluster.n_epnode=0 \
        cluster.n_ednode=0 \
        cluster.n_epdnode=$num_gpu \
        cluster.n_pnode=0 \
        cluster.n_pdnode=0 \
        cluster.n_dnode=0 \
        cluster.debug=false \
        cluster.epdnode.batch_scheduler.max_running_requests=24 \
        cluster.epdnode.request_processor.disaggregate_embed_prefill=true \
        cluster.epdnode.batch_scheduler.debug=False \
        cluster.epdnode.batch_scheduler.chunked_prefill=false \
        cluster.epdnode.batch_scheduler_profiler.profile_batch_config=false \
        cluster.epdnode.batch_scheduler.max_batch_fill_tokens=4096 \
        > $RESULT_PATH/${method}_api_server.log 2>&1 &
}


for method in "${methods[@]}"; do
    echo "Evaluating ${method}"
    num_gpu=1
    if [[ "$method" == "ours-abligation-batchpolicy" ]]; then
        start_batch_policy_abligation_study_server
        benchmark
        clean_up
        sleep 30
    elif [[ "$method" == ours* ]]; then
        if [ "$method" == "ours" ]; then
            multi_streams_forward=true
            multi_threads_forward=true
        elif [ "$method" == "ours-abligation-multistream" ]; then
            multi_streams_forward=false
            multi_threads_forward=false
        fi
        base_method=${method}
        disaggregation_methods=(
            "epd"
            "ep_d"
            "ed_p"
            "e_p_d"
        )
        num_gpu=8
        for scenario in "${scenarios[@]}"; do
            for disaggregation_method in "${disaggregation_methods[@]}"; do
                echo "disaggregation_method ${disaggregation_method}"
                jq -c ".${disaggregation_method}[\"$num_gpu\"] | .[]" "$disaggregation_methods_file" | while read n_node_config; do
                    n_enode=$(echo "$n_node_config" | jq '.n_enode')
                    n_epnode=$(echo "$n_node_config" | jq '.n_epnode')
                    n_ednode=$(echo "$n_node_config" | jq '.n_ednode')
                    n_pdnode=$(echo "$n_node_config" | jq '.n_pdnode')
                    n_epdnode=$(echo "$n_node_config" | jq '.n_epdnode')
                    n_pnode=$(echo "$n_node_config" | jq '.n_pnode')
                    n_dnode=$(echo "$n_node_config" | jq '.n_dnode')
                    if [[ "$disaggregation_method" == "epd" ]]; then
                        method="${base_method}-${n_epdnode}epd"
                    elif [[ "$disaggregation_method" == "ep_d" ]]; then
                        method="${base_method}-${n_epnode}ep${n_dnode}d"
                    elif [[ "$disaggregation_method" == "ed_p" ]]; then
                        method="${base_method}-${n_ednode}ed${n_pnode}p"
                    elif [[ "$disaggregation_method" == "e_p_d" ]]; then
                        method="${base_method}-${n_enode}e${n_pnode}p${n_dnode}d"
                    fi
                    echo $"Evaluating diaggregation methods ${method}"

                    max_retry=3
                    attempt=0
                    while [ $attempt -lt $max_retry ]; do
                        start_our_server
                        if benchmark_one_scenario "${scenario}"; then
                            echo "Benchmark success."
                            break
                        else
                            echo "Benchmark failed. Retrying..."
                            clean_up
                            attempt=$((attempt + 1))
                        fi
                    done
                    clean_up
                    sleep 30
                done
            done
        done
    elif [[ "$method" == vllm* ]]; then
        if [ "$method" == "vllm-tp8" ]; then
            VLLM_USE_V1=1 \
            conda run -n vllm --no-capture-output \
                vllm serve $MODEL_PATH \
                --host=$host \
                --port=$port \
                --tensor-parallel-size=8 \
                --enable-chunked-prefill \
                --no-enable-prefix-caching \
                --enforce-eager \
                --chat-template=$CHAT_TEMPLATE_PATH \
                > $RESULT_PATH/${method}_api_server.log 2>&1 &
            num_gpu=8
        elif [ "$method" == "vllm" ]; then
            VLLM_USE_V1=1 \
            conda run -n vllm --no-capture-output \
                vllm serve $MODEL_PATH \
                --host=$host \
                --port=$port \
                --enable-chunked-prefill \
                --no-enable-prefix-caching \
                --enforce-eager \
                --chat-template=$CHAT_TEMPLATE_PATH \
                > $RESULT_PATH/${method}_api_server.log 2>&1 &
        elif [ "$method" == "vllm-v0" ]; then
            VLLM_USE_V1=0 \
            conda run -n vllm --no-capture-output \
                vllm serve $MODEL_PATH \
                --host=$host \
                --port=$port \
                --no-enable-prefix-caching \
                --enforce-eager \
                --chat-template=$CHAT_TEMPLATE_PATH \
                --hf-overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}' \
                > $RESULT_PATH/${method}_api_server.log 2>&1 &
        elif [ "$method" == "vllm6" ]; then
            conda run -n vllm6 --no-capture-output \
                vllm serve $MODEL_PATH \
                --host=$host \
                --port=$port \
                --enforce-eager \
                --chat-template=$CHAT_TEMPLATE_PATH \
                > $RESULT_PATH/${method}_api_server.log 2>&1 &
        elif [ "$method" == "vllm7" ]; then
            conda run -n vllm7 --no-capture-output \
                vllm serve $MODEL_PATH \
                --host=$host \
                --port=$port \
                --no-enable-prefix-caching \
                --enforce-eager \
                --chat-template=$CHAT_TEMPLATE_PATH \
                > $RESULT_PATH/${method}_api_server.log 2>&1 &
        elif [ "$method" == "vllm8" ]; then
            conda run -n vllm8 --no-capture-output \
                vllm serve $MODEL_PATH \
                --host=$host \
                --port=$port \
                --no-enable-prefix-caching \
                --enforce-eager \
                --chat-template=$CHAT_TEMPLATE_PATH \
                > $RESULT_PATH/${method}_api_server.log 2>&1 &
        fi

        benchmark
    elif [ "$method" == "tgi" ]; then
        # # LD_LIBRARY_PATH is used to solve error text-generation-launcher: error while loading shared libraries: libpython3.11.so.1.0: cannot open shared object file: No such file or directory
        # PYTHON_PATH=$(conda run -n tgi python -c "import sys; print(sys.executable)")
        # ENV_DIR=$(dirname $(dirname "$PYTHON_PATH"))  # remove /bin/python
        # LIB_PATH="$ENV_DIR/lib"
        
        # RUST_BACKTRACE=1 LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH conda run -n tgi --no-capture-output \
        #     text-generation-launcher --model-id $MODEL_PATH \
        #     --hostname=$host \
        #     --port=$port \
        #     --payload-limit=200000000 \
        #     > $RESULT_PATH/${method}_api_server.log 2>&1 &
        benchmark
    elif [[ "$method" == sglang* ]]; then
        conda run -n ${method} --no-capture-output \
            python -m sglang.launch_server \
            --host=$host \
            --port=$port \
            --model-path=$MODEL_PATH \
            --chat-template=$CHAT_TEMPLATE_PATH \
            --disable-radix-cache \
            --disable-cuda-graph \
            --enable-multimodal \
            --disable-cuda-graph-padding \
            > $RESULT_PATH/${method}_api_server.log 2>&1 &
        benchmark
    elif [ "$method" == "lightllm" ]; then
        conda run -n ${method} --no-capture-output \
            python -m lightllm.server.api_server \
            --host=$host \
            --port=$port \
            --enable_multimodal \
            --model_dir ${MODEL_PATH} \
            > $RESULT_PATH/${method}_api_server.log 2>&1 &
        benchmark
    elif [ "$method" == "lmdeploy" ]; then
        conda run -n ${method} --no-capture-output \
            lmdeploy serve api_server \
            ${MODEL_PATH} \
            --server-port=${port} \
            > $RESULT_PATH/${method}_api_server.log 2>&1 &
        benchmark
    else
        echo "Unsupported method: $method"
        exit 1
    fi

    echo "Finished evaluating ${method}"
    clean_up
    sleep 20
done

# tgi apiserver docker launch command
# sudo docker rm -f tgi_llava_1_5_7b && sudo docker run --name tgi_llava_1_5_7b \
#     --gpus all --shm-size 2g -p 8899:80 \
#     -v /export/home/dongxianzhe1/projects/tgi/data:/data \
#     -v /mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf:/model \
#     ghcr.io/huggingface/text-generation-inference:3.2.3 \
# 	--payload-limit=200000000 \
# 	--cuda-graphs=0 \
# 	--model-id /model

# sudo docker rm -f tgi_llava_1_6_7b && sudo docker run --name tgi_llava_1_6_7b \
#     --gpus all --shm-size 2g -p 8899:80 \
#     -v /export/home/dongxianzhe1/projects/tgi/data:/data \
#     -v /mnt/cfs/9n-das-admin/llm_models/llava-v1.6-vicuna-7b-hf:/model \
#     ghcr.io/huggingface/text-generation-inference:3.2.3 \
# 	--payload-limit=200000000 \
# 	--cuda-graphs=0 \
# 	--model-id /model

# sudo docker rm -f tgi_qwenvl2_7b && sudo docker run --name tgi_qwenvl2_7b \
#     --gpus all --shm-size 2g -p 8899:80 \
#     -v /export/home/dongxianzhe1/projects/tgi/data:/data \
#     -v /mnt/cfs/9n-das-admin/llm_models/Qwen2-7B-Instruct:/model \
#     ghcr.io/huggingface/text-generation-inference:3.2.3 \
# 	--payload-limit=200000000 \
# 	--max-total-tokens=10000000 \
# 	--model-id /model