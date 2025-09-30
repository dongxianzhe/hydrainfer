mkdir -p /models

HF_ENDPOINTS=(
    "https://hf-mirror.com"
    ""
)

download_hf_model() {
    local model_name=$1
    local local_dir=$2

    for (( attempt=0; attempt < $(( ${#HF_ENDPOINTS[@]} * 3 )); attempt++ )); do
        HF_ENDPOINT=${HF_ENDPOINTS[$((attempt % ${#HF_ENDPOINTS[@]}))]}
        HF_ENDPOINT=$HF_ENDPOINT hf download "$model_name" --local-dir "$local_dir"
        
        if [ $? -eq 0 ]; then
            echo "Download $model_name to $local_dir successful!"
            break  
        else
            echo "Download failed, attempt: $((attempt + 1))"
        fi
    done
    
    if [ $attempt -ge $(( ${#HF_ENDPOINTS[@]} * 3 )) ]; then
        echo "Download failed, maximum retry attempts reached!"
    fi
}

download_hf_model "llava-hf/llava-1.5-7b-hf" "/models/llava-hf/llava-1.5-7b-hf"
download_hf_model "llava-hf/llava-1.5-13b-hf" "/models/llava-hf/llava-1.5-13b-hf"
download_hf_model "llava-hf/llava-v1.6-vicuna-7b-hf" "/models/llava-hf/llava-v1.6-vicuna-7b-hf"
download_hf_model "llava-hf/llava-v1.6-vicuna-13b-hf" "/models/llava-hf/llava-v1.6-vicuna-13b-hf"
download_hf_model "Qwen/Qwen2-VL-7B" "/models/Qwen/Qwen2-VL-7B"
download_hf_model "deepseek-ai/deepseek-vl2-tiny" "/models/deepseek-ai/deepseek-vl2-tiny"
download_hf_model "OpenGVLab/InternVL2-26B" "/models/OpenGVLab/InternVL2-26B"