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

download_hf_dataset "lmms-lab/TextCaps" "/datasets/lmms-lab/TextCaps"
download_hf_dataset "lmms-lab/POPE" "/datasets/lmms-lab/POPE"
download_hf_dataset "lmms-lab/MME" "/datasets/lmms-lab/MME"
download_hf_dataset "lmms-lab/textvqa" "/datasets/lmms-lab/textvqa"
download_hf_dataset "lmms-lab/VizWiz-VQA" "/datasets/lmms-lab/VizWiz-VQA"