mkdir -p /datasets

HF_ENDPOINTS=(
    "https://hf-mirror.com"
    ""
)

download_hf_dataset() {
    local dataset_name=$1
    local local_dir=$2

    for (( attempt=0; attempt < $(( ${#HF_ENDPOINTS[@]} * 3 )); attempt++ )); do
        HF_ENDPOINT=${HF_ENDPOINTS[$((attempt % ${#HF_ENDPOINTS[@]}))]}
        HF_ENDPOINT=$HF_ENDPOINT hf download --repo-type dataset "$dataset_name" --local-dir "$local_dir"
        
        if [ $? -eq 0 ]; then
            echo "Download $dataset_name to $local_dir successful!"
            break  
        else
            echo "Download failed, attempt: $((attempt + 1))"
        fi
    done
    
    if [ $attempt -ge $(( ${#HF_ENDPOINTS[@]} * 3 )) ]; then
        echo "Download failed, maximum retry attempts reached!"
    fi
}

download_dataset_if_not_exists() {
    # Arguments: $1 - dataset name, $2 - dataset path
    local DATASET=$1
    local DATASET_PATH=$2

    if [ -d "$DATASET_PATH" ]; then
        echo "DATASET ${DATASET} exists in ${DATASET_PATH}, skipping download of HF dataset."
    else
        echo "DATASET ${DATASET} does not exist, attempting to download HF dataset to ${DATASET_PATH}."
        download_hf_dataset "$DATASET" "$DATASET_PATH"
    fi
}

main() {
  if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_name> <destination>"
    exit 1
  fi

  local dataset_name=$1
  local destination=$2

  download_dataset_if_not_exists "$dataset_name" "$destination"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi