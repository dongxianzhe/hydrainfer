import torch
from transformers import LlavaConfig, AutoProcessor, AutoTokenizer
from transformers import LlavaForConditionalGeneration as LlavaForConditionalGenerationRef
from dxz.model.llava import LlavaForConditionalGeneration
from dxz.model.clip import CLIPVisionModel
from dxz.memory.kv_cache import KVCache
from dxz.model.parameters import InputParameters
from dxz.model.downloader import download_hf_model
from PIL import Image


def test_llava1_5_7b_hf():
    # 1. config
    model_name = "llava-hf/llava-1.5-7b-hf"
    dtype = torch.half
    device = torch.device('cuda:0')

    # 2. download from hugging face
    model_path = download_hf_model(repo_id=model_name)
    
    # 3. create model from safe tensor
    model = LlavaForConditionalGeneration.from_safetensor(model_path, dtype, device)

    # model_ref = LlavaForConditionalGenerationRef.from_pretrained(
    #     pretrained_model_name_or_path=model_name_or_path, device_map=device, torch_dtype=torch.float
    # )

    # 4. prepare input and output
    text = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
    images = Image.open("/home/xzd/projects/dxz/learn/llava/panda.png")
    processor_ref = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)

    inputs = processor_ref(
        text=text, 
        images=images
    )
    token_ids = inputs.input_ids[0]
    print(f'len(token_ids) {len(token_ids)}')
    def insert_image_tokens(token_ids: list[int]) -> list[int]:
        inserted_token_ids = []
        for token_id in token_ids:
            if token_id == model.config.image_token_index:
                inserted_token_ids.extend([model.config.image_token_index] * (model.get_num_image_token_ids() - 1))
            inserted_token_ids.append(token_id)
        return inserted_token_ids
    token_ids = insert_image_tokens(token_ids)
    print(f'len(token_ids) {len(token_ids)}')

    pixel_values = inputs.pixel_values 
    pixel_values = torch.tensor(pixel_values, dtype=dtype, device=device)
    n_prompt_tokens = len(token_ids)
    n_kv_cache_tokens = 0
    output_ids = []

    n_blocks = 100
    block_size = 16
    num_key_value_heads = model.config.text_config.num_key_value_heads
    head_dim = model.config.text_config.head_dim
    n_layers = model.config.text_config.num_hidden_layers
    kv_caches = [KVCache(n_blocks, block_size, num_key_value_heads, head_dim, dtype=dtype, device=device) for _ in range(n_layers)]

    # 5. prefill
    input_ids = torch.tensor(token_ids, dtype=torch.int, device=device)
    pixel_values = torch.tensor(pixel_values, dtype=dtype, device=device)
    position_ids = torch.arange(len(token_ids), dtype=torch.int ,device=device)
    input_params = InputParameters(
        num_sequences = 1,
        q_cu_seq_lens = torch.tensor([0, n_prompt_tokens], dtype=torch.int, device=device), 
        kv_cu_seq_lens = torch.tensor([0, n_prompt_tokens], dtype=torch.int, device=device), 
        new_cache_slots = torch.arange(n_prompt_tokens, dtype=torch.int, device=device), 
        block_tables = torch.arange(n_blocks, dtype=torch.int, device=device), 
        cu_blocks_lens = torch.tensor([0, n_blocks], dtype=torch.int, device=device), 
        q_max_seq_len = 1024, 
        kv_max_seq_len = 1024
    )

    logit = model(input_ids, pixel_values, position_ids, kv_caches, input_params)[-1]
    n_kv_cache_tokens = n_prompt_tokens
    next_token_id = torch.argmax(logit, dim=-1).item()
    output_ids.append(next_token_id)

    # 5. decode
    max_new_tokens = 15
    for i in range(max_new_tokens):
        input_ids = torch.tensor([next_token_id], dtype=torch.int, device=device)
        pixel_values = None
        position_ids = torch.tensor([n_kv_cache_tokens], dtype=torch.int, device=device)
        input_params = InputParameters(
            num_sequences = 1,
            q_cu_seq_lens = torch.tensor([0, 1], dtype=torch.int, device=device), 
            kv_cu_seq_lens = torch.tensor([0, n_kv_cache_tokens + 1], dtype=torch.int, device=device), 
            new_cache_slots = torch.tensor([n_kv_cache_tokens], dtype=torch.int, device=device), 
            block_tables = torch.arange(n_blocks, dtype=torch.int, device=device), 
            cu_blocks_lens = torch.tensor([0, n_blocks], dtype=torch.int, device=device), 
            q_max_seq_len = 1024, 
            kv_max_seq_len = 1024
        )
        logit = model(input_ids, pixel_values, position_ids, kv_caches, input_params)[-1]
        n_kv_cache_tokens += 1

        next_token_id = torch.argmax(logit, dim=-1).item()
        output_ids.append(next_token_id)

    output_text = tokenizer.decode(output_ids)
    print(f'output_text: {output_text}')

if __name__ == '__main__':
    test_llava1_5_7b_hf()