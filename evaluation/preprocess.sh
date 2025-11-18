#!/bin/bash 
source download_dataset.sh
source common.sh
source config.sh

############################## params ##############################
model_used_in_preprocess="llava-hf/llava-1.5-7b-hf"
####################################################################

preprocess_dataset(){
    local dataset=$1
    local dataset_path=$2
    
    TOKENIZERS_PARALLELISM=true \
    CUDA_VISIBLE_DEVICES=1 conda run -n vllm-0-11-0 --no-capture-output \
        python ${OUR_ROOT_PATH}/benchmark/data_preprocess.py \
            --dataset=${dataset} \
            --dataset-path=${dataset_path} \
            --model=${model_used_in_preprocess} \
            --model-path=${model_to_path[$model_used_in_preprocess]}
}

preprocess_dataset_if_necessary(){
    # Arguments: $1 - dataset_name name, $2 - dataset_name path
    local DATASET=$1
    local DATASET_PATH=$2

    download_dataset_if_not_exists $DATASET $DATASET_PATH
    PREPROCESSED_DATASET_PATH="$DATASET_PATH/synthetic_source_dataset.json"

    if [ -f "$PREPROCESSED_DATASET_PATH" ]; then
        echo "preprocessed DATASET ${DATASET} exists in ${PREPROCESSED_DATASET_PATH}."
    else
        echo "preprocessed DATASET ${DATASET} does not exist, attempting to preprocess HF dataset to ${PREPROCESSED_DATASET_PATH}."
        preprocess_dataset "$DATASET" "$DATASET_PATH"
    fi
}

main() {
  if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_name> <destination>"
    exit 1
  fi

  local dataset_name=$1
  local destination=$2

  preprocess_dataset_if_necessary "$dataset_name" "$destination"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi