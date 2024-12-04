from itertools import accumulate
from PIL import Image
import torch
from transformers import AutoProcessor
from dxz.model.llava import LlavaForConditionalGeneration
from dxz.model.downloader import download_hf_model
from dxz.model.parameters import ModelParameters, AttentionParameters
from dxz.memory.kv_cache import KVCache
from uitl import bar_chart
import time

dtype = torch.half
device = torch.device('cuda:0')
model_name = "llava-hf/llava-1.5-7b-hf"
model_path = download_hf_model(repo_id=model_name)
processor = AutoProcessor.from_pretrained(model_path)
tokenizer = processor.tokenizer
model = LlavaForConditionalGeneration.from_safetensor(model_path, dtype, device)
model_config = model.config
num_kv_heads = model_config.text_config.num_key_value_heads
head_size = model_config.text_config.head_dim
n_layers = model_config.text_config.num_hidden_layers


# 1. test prefill
def prefill_batchsize_throughput(batch_size) -> float:
    prompt_length = 512
    num_sequences = batch_size
    total_blocks = 2048
    block_size = 16
    n_tokens = prompt_length * num_sequences
    input_ids = torch.zeros(size=(n_tokens, ), dtype=torch.int, device=device)
    position_ids = torch.zeros(size=(n_tokens, ), dtype=torch.int, device=device)
    kv_cache = KVCache(total_blocks, block_size, num_kv_heads, head_size, dtype, device)
    q_seq_lens = [prompt_length for _ in range(num_sequences)]
    q_cu_seq_lens = torch.tensor(list(accumulate(q_seq_lens, initial=0)), dtype=torch.int, device=device)
    k_seq_lens = q_seq_lens
    kv_cu_seq_lens = torch.tensor(list(accumulate(k_seq_lens, initial=0)), dtype=torch.int, device=device)
    new_cache_slots = torch.arange(prompt_length, dtype=torch.int, device=device)[None, :].repeat(num_sequences, 1).reshape(-1)

    n_blocks = (prompt_length + block_size - 1) // block_size
    block_lens = [n_blocks for _ in range(num_sequences)]
    block_tables = torch.arange(n_blocks, dtype=torch.int, device=device)[None, :].repeat(num_sequences, 1).reshape(-1)
    cu_blocks_lens = torch.tensor(list(accumulate(block_lens, initial=0)), dtype=torch.int, device=device)
    q_max_seq_len = max(q_seq_lens)
    kv_max_seq_len = max(k_seq_lens)

    model_params = ModelParameters(
        attention_params = [AttentionParameters(
            kv_cache=kv_cache,
            q_cu_seq_lens = q_cu_seq_lens, 
            kv_cu_seq_lens = kv_cu_seq_lens, 
            new_cache_slots = new_cache_slots, 
            block_tables = block_tables, 
            cu_blocks_lens = cu_blocks_lens, 
            num_sequences = num_sequences, 
            all_sequences_decode = False, 
            q_max_seq_len = q_max_seq_len,
            kv_max_seq_len = kv_max_seq_len, 
        )for layer_id in range(n_layers)]
    )
    logits = model(input_ids=input_ids, pixel_values=None, image_features=None, position_ids=position_ids, model_params=model_params)
    logits = model(input_ids=input_ids, pixel_values=None, image_features=None, position_ids=position_ids, model_params=model_params)
    logits = model(input_ids=input_ids, pixel_values=None, image_features=None, position_ids=position_ids, model_params=model_params)

    start = time.perf_counter()
    num_repeats = 10
    for i in range(num_repeats):
        logits = model(input_ids=input_ids, pixel_values=None, image_features=None, position_ids=position_ids, model_params=model_params)
        torch.cuda.synchronize()
    end = time.perf_counter()
    duration = end - start
    throughput = n_tokens * num_repeats / duration
    return throughput

def plot_prefill_batchsize_throughput():
    batch_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
    throughputs = []
    for batch_size in batch_sizes:
        throughput = prefill_batchsize_throughput(batch_size)
        print(f"batch_size {batch_size} throughput {throughput} tokens / s")
        throughputs.append(throughput)

    bar_chart(batch_sizes, throughputs, figsize=(8, 6), xlabel="Batch Size", ylabel="Tokens per Second", title="Prefill", filename="bs2throughput")

# 2. test decode
def decode_batchsize_throughput(batch_size) -> float:
    prompt_length = 512
    num_sequences = batch_size
    total_blocks = 2048
    block_size = 16
    n_tokens = 1 * num_sequences
    input_ids = torch.zeros(size=(n_tokens, ), dtype=torch.int, device=device)
    position_ids = torch.zeros(size=(n_tokens, ), dtype=torch.int, device=device)
    kv_cache = KVCache(total_blocks, block_size, num_kv_heads, head_size, dtype, device)
    q_seq_lens = [1 for _ in range(num_sequences)]
    q_cu_seq_lens = torch.tensor(list(accumulate(q_seq_lens, initial=0)), dtype=torch.int, device=device)
    k_seq_lens = [prompt_length for _ in range(num_sequences)]
    kv_cu_seq_lens = torch.tensor(list(accumulate(k_seq_lens, initial=0)), dtype=torch.int, device=device)
    new_cache_slots = torch.arange(num_sequences, dtype=torch.int, device=device)

    n_blocks = (prompt_length + block_size - 1) // block_size
    block_lens = [n_blocks for _ in range(num_sequences)]
    block_tables = torch.arange(n_blocks, dtype=torch.int, device=device)[None, :].repeat(num_sequences, 1).reshape(-1)
    cu_blocks_lens = torch.tensor(list(accumulate(block_lens, initial=0)), dtype=torch.int, device=device)
    q_max_seq_len = max(q_seq_lens)
    kv_max_seq_len = max(k_seq_lens)
    model_params = ModelParameters(
        attention_params = [AttentionParameters(
            kv_cache=kv_cache,
            q_cu_seq_lens = q_cu_seq_lens, 
            kv_cu_seq_lens = kv_cu_seq_lens, 
            new_cache_slots = new_cache_slots, 
            block_tables = block_tables, 
            cu_blocks_lens = cu_blocks_lens, 
            num_sequences = num_sequences, 
            all_sequences_decode = False, 
            q_max_seq_len = q_max_seq_len,
            kv_max_seq_len = kv_max_seq_len, 
        )for layer_id in range(n_layers)]
    )
    logits = model(input_ids=input_ids, pixel_values=None, image_features=None, position_ids=position_ids, model_params=model_params)
    logits = model(input_ids=input_ids, pixel_values=None, image_features=None, position_ids=position_ids, model_params=model_params)
    logits = model(input_ids=input_ids, pixel_values=None, image_features=None, position_ids=position_ids, model_params=model_params)

    start = time.perf_counter()
    num_repeats = 10
    for i in range(num_repeats):
        logits = model(input_ids=input_ids, pixel_values=None, image_features=None, position_ids=position_ids, model_params=model_params)
        torch.cuda.synchronize()
    end = time.perf_counter()
    duration = end - start
    throughput = num_sequences * num_repeats / duration
    return throughput 

def plot_decode_batchsize_throughput():
    batch_sizes = [i for i in range(1, 64)]
    throughputs = []
    for batch_size in batch_sizes:
        throughput = decode_batchsize_throughput(batch_size)
        print(f"batch_size {batch_size} throughput {throughput} tokens / s")
        throughputs.append(throughput)

    bar_chart(batch_sizes, throughputs, figsize=(8, 6), xlabel="Batch Size", ylabel="Tokens per Second", title="Decode", filename="decodebs2throughput")

def embed_batchsize_throughput(batch_size) -> float:
    model_params = ModelParameters()
    pixel_values = torch.randn(batch_size, 3, 336, 336, dtype=torch.half, device=device)

    image_features = model.image_embed(pixel_values, model_params)
    image_features = model.image_embed(pixel_values, model_params)
    image_features = model.image_embed(pixel_values, model_params)

    start = time.perf_counter()
    num_repeats = 10
    for i in range(num_repeats):
        image_features = model.image_embed(pixel_values, model_params)
        print(image_features.shape)
        torch.cuda.synchronize()
    end = time.perf_counter()
    duration = end - start
    throughput = 576 * batch_size *  num_repeats / duration
    return throughput

def plot_embed_batchsize_throughput():
    batch_sizes = [i for i in range(1, 8)]
    throughputs = []
    for batch_size in batch_sizes:
        throughput = embed_batchsize_throughput(batch_size)
        print(f"batch_size {batch_size} throughput {throughput} tokens/s")
        throughputs.append(throughput)

    bar_chart(batch_sizes, throughputs, figsize=(8, 6), xlabel="Batch Size", ylabel="Tokens per Second", title="Embed", filename="embed_bs2throughput")

# 3. test embed
if __name__ == '__main__':
    # plot_prefill_batchsize_throughput()
    # plot_decode_batchsize_throughput()
    plot_embed_batchsize_throughput()