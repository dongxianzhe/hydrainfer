dataset_root_path="/datasets"
model_root_path="/models"
trace_root_path="/trace"

declare -A dataset_to_path=(
    ["lmms-lab/MME"]="/datasets/lmms-lab/MME"
    ["lmms-lab/TextCaps"]="/datasets/lmms-lab/TextCaps"
    ["lmms-lab/POPE"]="/datasets/lmms-lab/POPE"
    ["lmms-lab/textvqa"]="/datasets/lmms-lab/textvqa"
    ["lmms-lab/VizWiz-VQA"]="/datasets/lmms-lab/VizWiz-VQA"
)

declare -A model_to_path=(
    ["llava-hf/llava-1.5-7b-hf"]="$model_root_path/llava-hf/llava-1.5-7b-hf"
    ["llava-hf/llava-v1.6-vicuna-7b-hf"]="$/model_root_path/llava-hf/llava-v1.6-vicuna-7b-hf"
    ["Qwen/Qwen2-VL-7B"]="$model_root_path/Qwen/Qwen2-VL-7B"
    ["deepseek-ai/deepseek-vl2-tiny"]="$model_root_path/deepseek-ai/deepseek-vl2-tiny"
    ["OpenGVLab/InternVL2-26B"]="$model_root_path/OpenGVLab/InternVL2-26B"
)
