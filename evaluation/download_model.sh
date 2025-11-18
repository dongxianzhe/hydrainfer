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

download_model_if_not_exists() {
    # Arguments: $1 - model name, $2 - model path
    local MODEL=$1
    local MODEL_PATH=$2

    if [ -d "$MODEL_PATH" ]; then
        echo "Model ${MODEL} exists in ${MODEL_PATH}, skipping download of HF model."
    else
        echo "Model ${MODEL} does not exist, attempting to download HF model to ${MODEL_PATH}."
        download_hf_model "$MODEL" "$MODEL_PATH"
    fi
}

main() {
  if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_name> <destination>"
    exit 1
  fi

  local model_name=$1
  local destination=$2

  download_model_if_not_exists "$model_name" "$destination"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi