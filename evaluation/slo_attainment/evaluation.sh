#!/bin/bash 

source ../common.sh

export CUDA_VISIBLE_DEVICES=2
MODEL="llava-hf/llava-1.5-7b-hf"
# MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf"
MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/llava-v1.6-vicuna-7b-hf"
# REQUEST_RATES="1 2 3 4 5 6 7 8 9 10"
# NUM_REQUESTS=200
REQUEST_RATES="5"
NUM_REQUESTS=50
host="127.0.0.1"
port="8888"

scenarios=(
    "--textcaps=1 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=0"
    # "--textcaps=0 --pope=1 --mme=0 --text_vqa=0 --vizwiz_vqa=0"
    # "--textcaps=0 --pope=0 --mme=1 --text_vqa=0 --vizwiz_vqa=0"
    # "--textcaps=0 --pope=0 --mme=0 --text_vqa=1 --vizwiz_vqa=0"
    # "--textcaps=0 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=1"
)

methods=(
    "tgi"
    # "sglang"
    # "llama"
    "vllm"
    # "ours"
)

for method in "${methods[@]}"; do
    echo "Evaluating ${method}"

    if [ "$method" == "vllm" ]; then
        conda run -n vllm --no-capture-output \
            vllm serve $MODEL_PATH \
            --host=$host \
            --port=$port \
            --enable-chunked-prefill \
            --no-enable-prefix-caching \
            --enforce-eager \
            --chat-template=$OUR_ROOT_PATH/dxz/chat_template/template_llava.jinja \
            > $RESULT_PATH/${method}_api_server.log 2>&1 &
    elif [ "$method" == "ours" ]; then
        RAY_DEDUP_LOGS=0 \
            conda run -n dxz_dev --no-capture-output \
            python -m dxz.entrypoint.entrypoint \
            model.name=$MODEL \
            model.path=$MODEL_PATH \
            cluster=single \
            apiserver.host=$host \
            apiserver.port=$port \
            > $RESULT_PATH/${method}_api_server.log 2>&1 &
    elif [ "$method" == "tgi" ]; then
        # LD_LIBRARY_PATH is used to solve error text-generation-launcher: error while loading shared libraries: libpython3.11.so.1.0: cannot open shared object file: No such file or directory
        PYTHON_PATH=$(conda run -n tgi python -c "import sys; print(sys.executable)")
        ENV_DIR=$(dirname $(dirname "$PYTHON_PATH"))  # remove /bin/python
        LIB_PATH="$ENV_DIR/lib"
        
        RUST_BACKTRACE=1 LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH conda run -n tgi --no-capture-output \
            text-generation-launcher --model-id $MODEL_PATH \
            --hostname=$host \
            --port=$port \
            --payload-limit=200000000 \
            > $RESULT_PATH/${method}_api_server.log 2>&1 &
    else
        echo "Unsupported method: $method"
        exit 1
    fi

    wait_api_server $host $port

    echo "Start benchmarking"
    for scenario in "${scenarios[@]}"; do
        echo "Running scenario: $scenario"
        conda run -n dxz_dev --no-capture-output \
            python $OUR_ROOT_PATH/benchmark/benchmark.py \
            --num-requests=$NUM_REQUESTS \
            --model=$MODEL \
            --model-path=$MODEL_PATH \
            --method-name=${method} \
            --request-rate ${REQUEST_RATES} \
            --backend=$method \
            $scenario \
            --result-path="$RESULT_PATH/${method}_${scenario// /_}_results.json"
    done

    echo "Finished evaluating ${method}"
    clean_up
    sleep 20
done