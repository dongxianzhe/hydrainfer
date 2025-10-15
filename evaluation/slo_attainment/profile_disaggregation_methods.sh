source ../common.sh

declare -A MODELS=(
    # ["llava-hf/llava-1.5-7b-hf"]="/models/llava-1.5-7b-hf"
    ["llava-hf/llava-v1.6-vicuna-7b-hf"]="/models/llava-v1.6-vicuna-7b-hf"
    # ["Qwen/Qwen2-VL-7B"]="/models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024"
    # ["deepseek-ai/deepseek-vl2-tiny"]="/models/deepseek-vl2-tiny"
    # ["OpenGVLab/InternVL2-26B"]="/models/OpenGVLab/InternVL2-26B"
)
trace_configs=(
    "--textcaps=1 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=0"
    # "--textcaps=0 --pope=1 --mme=0 --text_vqa=0 --vizwiz_vqa=0"
    # "--textcaps=0 --pope=0 --mme=1 --text_vqa=0 --vizwiz_vqa=0"
    # "--textcaps=0 --pope=0 --mme=0 --text_vqa=1 --vizwiz_vqa=0"
    # "--textcaps=0 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=1"
)
for MODEL in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL]}"
    for trace in "${trace_configs[@]}"; do
        echo "==================== ${MODEL} ${trace} =============================="
        conda run -n hydrainfer --no-capture-output \
            python ${OUR_ROOT_PATH}/benchmark/workload_profiler.py \
                --model-path=$MODEL_PATH \
                $trace \
                --ttft_slo=8 \
                --tpot_slo=0.4
    done
done
