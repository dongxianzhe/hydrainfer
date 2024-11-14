import os
import copy
import csv
import torch
import time
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import seaborn as sns
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
import warnings
warnings.filterwarnings("ignore")

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Pretrained model settings
pretrained = "../models/liuhaotian/llava-v1.5-7b"
model_name = "llava_llama"
device = "cuda"
device_map = "cuda:0"
load_4bit = False

# Load model and tokenizer once
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device_map, load_4bit=load_4bit
)
model.eval()
model.tie_weights()

# Example image to use across multiple inferences
# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("llava_v1_5_radar.jpg")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

# Preprocess question without conversation context
def preprocess_question(question):
    input_text = DEFAULT_IMAGE_TOKEN + "\n" + question  # Add image token before question
    # input_text = question + "\n" + DEFAULT_IMAGE_TOKEN
    input_ids = tokenizer_image_token(input_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    return input_ids

# Run inference without reloading the model
def run_inference(question):
    input_ids = preprocess_question(question)

    # Measure GPU memory and time
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    first_token_time = None

    # Hook to capture the time for the first token
    def hook_function(module, input, output):
        nonlocal first_token_time
        if first_token_time is None:
            first_token_time = time.time()

    handle = model.lm_head.register_forward_hook(hook_function)

    time_start = time.time()
    output_ids = model(
        input_ids,
        images=image_tensor,          # Image tensor as input
        image_sizes=[image.size],     # Image size
        output_attentions=True,
        use_cache=True,
    )
    time_end = time.time()
    handle.remove()

    # Calculate GPU memory and time after generation
    torch.cuda.synchronize(device)
    max_gpu_memory_cost = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # Convert to GB

    # Decode text outputs
    # text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Calculate Time to First Token (TTFT)
    if first_token_time is not None:
        ttft = first_token_time - time_start
    else:
        ttft = "N/A"

    # Calculate average time per token
    # num_tokens_generated = output_ids.shape[1]
    # if num_tokens_generated > 0:
    #     remaining_tokens_time = time_end - time_start - ttft
    #     avg_time_per_token = remaining_tokens_time / (num_tokens_generated-1)
    # else:
    #     avg_time_per_token = "N/A"
    attentions = output_ids.attentions
    print(f"len(attention):{len(attentions)}\tshape(attention[0]):{attentions[0].shape}\t{attentions[0][0].mean(dim=0).shape}")
    for layer_index, attention in enumerate(attentions):
        # # 选择第一个样本的 attention map
        # attention_map = attention[0].mean(dim=0).detach().cpu().numpy()
        # # 可视化并保存为 PNG
        # plt.figure(figsize=(10, 10))
        # plt.imshow(attention_map, cmap='Reds', vmin=0, vmax=0.01)
        # plt.colorbar()
        # plt.title("Attention Map")
        # plt.xlabel("Token Position")
        # plt.ylabel("Token Position")
        # # # 保存为 PNG 文件
        # plt.savefig(f'attention_txt_img/attention_layer_{layer_index + 1}.png')
        # plt.close()  # 关闭当前图形，释放内存
        averaged_attention = torch.mean(attention, axis=1)[0].float().cpu().detach().numpy() # Shape: (n_tokens, n_tokens)
        cmap = plt.cm.get_cmap("Reds")
        plt.figure(figsize=(5, 5))
    
        # Log normalization
        log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())
    
        # set the x and y ticks to 20x of the original
    
    
        ax = sns.heatmap(averaged_attention,
                    cmap=cmap,  # custom color map
                    norm=log_norm,  # 
                    # cbar_kws={'label': 'Attention score'},
                    )
        
        # remove the x and y ticks
        
        # replace the x and y ticks with string
    
        # change the x tinks font size
        plt.xticks(fontsize=3)
        plt.yticks(fontsize=3)
        
        # make y label vertical
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)     
    
        # tight layout
        plt.savefig(f'attention/attention_layer_{layer_index + 1}.png', bbox_inches='tight')
        plt.close()
    print("Attention maps saved as PNG files.")
        
    return {
        # "text_outputs": text_outputs,
        "ttft": ttft,
        # "avg_time_per_token": avg_time_per_token,
        "gpu_memory_cost": max_gpu_memory_cost,
        "elapsed_time": time_end - time_start
    }

def save_results_to_csv(results, csv_file, ratio):
    # Open or create the CSV file to store the results
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row with proper column names, including the `ratio`
        writer.writerow(["Run", "TTFT (seconds)", "Avg Time per Token (seconds)", "GPU Memory Cost (GB)", "Ratio"])
        
        # Write the accumulated results to the CSV file
        for i, result in enumerate(results):
            writer.writerow([i + 1, result['ttft'], result['avg_time_per_token'], result['gpu_memory_cost'], ratio])

    print(f"Results saved to {csv_file}")

# List to store the results
all_results = []

# Set your ratio parameter (you can specify this value as needed)
ratio = 1.00

# Run inference 10 times and accumulate the results
question = "Describe this picture?"
results = run_inference(question)
# for i in range(10):
#     results = run_inference(question)
    
#     # Save each result to the list
#     all_results.append({
#         'ttft': results['ttft'],
#         'avg_time_per_token': results['avg_time_per_token'],
#         'gpu_memory_cost': results['gpu_memory_cost']
#     })
    
#     # Print the result to the console
#     print(f"Run {i+1}: TTFT={results['ttft']}s, Avg time per token={results['avg_time_per_token']}s, GPU memory cost={results['gpu_memory_cost']}GB")
#     # print(results['text_outputs'])

# # Specify the CSV file path
# csv_file = 'inference_results_with_ratio_1.5.csv'

# # Save all the accumulated results to the CSV file with the ratio
# save_results_to_csv(all_results, csv_file, ratio)