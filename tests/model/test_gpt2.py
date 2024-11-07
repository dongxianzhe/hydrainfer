import torch
from transformers import GPT2Config, GPT2Tokenizer
from dxz.model.gpt2 import GPT2LMHeadModel
from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
from dxz.utils.model_utils import count_parameters, print_parameters
from dxz.model.parameters import InputParameters
from dxz.memory.kv_cache import KVCache
import pytest

def test_params():
    model_ref = GPT2LMHeadModelRef.from_pretrained('gpt2')
    model = GPT2LMHeadModel(model_ref.config)
    param_count = count_parameters(model)
    param_count_ref = count_parameters(model_ref)
    assert param_count == param_count_ref, f'{param_count} {param_count_ref}'

def test_forward():
    device = torch.device('cuda:0')
    dtype=torch.half

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model_ref = GPT2LMHeadModelRef.from_pretrained('gpt2')
    model     = GPT2LMHeadModel.from_pretrained('gpt2')
    model_ref.to(dtype)
    model_ref.to(device)
    model_ref.eval()
    model.to(dtype)
    model.to(device)
    model.eval()
    config = model_ref.config

    prompt = "Okay, let's rewrite this. We need some conversation between Jack and his mother immediately after - perhaps over dinner back at their house? She is concerned, but not doubting him too much, and he has to convince her to let him come back to the museum?"

    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(token_ids, dtype=torch.int, device=device)
    position_ids = torch.arange(len(token_ids), dtype=torch.int, device=device)
    logits_ref = model_ref(input_ids=input_ids, position_ids=position_ids)['logits']
    next_token_id_ref = torch.argmax(logits_ref[-1, :])

    kv_caches = [KVCache(100, 16, config.n_head, config.n_embd // config.n_head, dtype, device) for _ in range(config.n_layer)]

    input_params = InputParameters(
        num_sequences = 1, 
        q_cu_seq_lens = torch.tensor([0, len(token_ids)], dtype=torch.int ,device=device),
        kv_cu_seq_lens = torch.tensor([0, len(token_ids)], dtype=torch.int ,device=device),
        new_cache_slots = torch.arange(len(token_ids), dtype=torch.int, device=device),
        block_tables = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int, device=device), 
        cu_blocks_lens = torch.tensor([0, 8], dtype=torch.int, device=device)
    )
    logits = model(input_ids, position_ids, kv_caches, input_params)
    next_token_id = torch.argmax(logits[-1, :])

    print(f'next_token_id {next_token_id} {tokenizer.decode(next_token_id)}')
    print(f'next_token_id_ref {next_token_id_ref} {tokenizer.decode(next_token_id_ref)}')
    print(f'new_cache_slots {input_params.new_cache_slots}')

    print(f'{logits.shape} {logits[-1, :10]}')
    print(f'{logits_ref.shape} {logits_ref[-1, :10]}')
    assert torch.allclose(logits, logits_ref, atol=1, rtol=1e-2), 'forward output wrong'

def test_generate():
    device = torch.device('cuda:0')
    dtype=torch.half
    max_tokens = 50

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model     = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(dtype)
    model.to(device)
    model.eval()
    config = model.config

    prompt = "Okay, let's rewrite this. We need some conversation between Jack and his mother immediately after - perhaps over dinner back at their house? She is concerned, but not doubting him too much, and he has to convince her to let him come back to the museum?"
    output_text_ref = "He is not going to let her go, and he is not going to let her go. He is going to let her go, and he is going to let her go. He is going to let her go, and he is going to let"

    kv_caches = [KVCache(100, 16, config.n_head, config.n_embd // config.n_head, dtype=dtype, device=device)for _ in range(config.n_layer)]
    token_ids = tokenizer.encode(prompt)
    n_prompt_tokens = len(token_ids)
    # prefill
    input_ids = torch.tensor(token_ids, dtype=torch.int, device=device)
    position_ids = torch.arange(len(token_ids), dtype=torch.int, device=device)
    input_params = InputParameters(
        num_sequences = 1, 
        q_cu_seq_lens = torch.tensor([0, len(token_ids)], dtype=torch.int ,device=device),
        kv_cu_seq_lens = torch.tensor([0, len(token_ids)], dtype=torch.int ,device=device),
        new_cache_slots = torch.arange(len(token_ids), dtype=torch.int, device=device),
        block_tables = torch.arange(10, dtype=torch.int, device=device), 
        cu_blocks_lens = torch.tensor([0, 10], dtype=torch.int, device=device)
    )
    logits = model(input_ids, position_ids, kv_caches, input_params)
    next_token_id = torch.argmax(logits[-1, :])
    token_ids.append(next_token_id)
    print(f'============================= prefill ===============================')
    print(f'new_cache_slots {input_params.new_cache_slots}')
    # decode
    for i in range(max_tokens - 1):
        position_id = len(token_ids) - 1
        input_ids = torch.tensor(next_token_id, dtype=torch.int ,device=device)
        position_ids = torch.tensor([position_id], dtype=torch.int, device=device)
        input_params = InputParameters(
            num_sequences = 1, 
            q_cu_seq_lens = torch.tensor([0, 1], dtype=torch.int, device=device), 
            kv_cu_seq_lens = torch.tensor([0, len(token_ids)], dtype=torch.int, device=device), 
            new_cache_slots = torch.tensor([position_id], dtype=torch.int, device=device), 
            block_tables = torch.arange(10, dtype=torch.int ,device=device), 
            cu_blocks_lens = torch.tensor([0, 10], dtype=torch.int ,device=device)
        )
        logits = model(input_ids, position_ids, kv_caches, input_params)
        next_token_id = torch.argmax(logits[-1, :], dim=-1).item()
        if next_token_id == tokenizer.eos_token_id:
            break
        token_ids.append(next_token_id)
        print(f'=============================decode {i}===============================')
        print(f'new_cache_slots {input_params.new_cache_slots}')
        print(f'next_token_id {next_token_id} {tokenizer.decode(next_token_id)}')
        if i == 25:
            and_token_id = tokenizer.encode([','])[0]
            print(logits[-1, and_token_id], logits[-1, next_token_id])

    print(f'=============================end===============================')
    print(f'len(token_ids) {len(token_ids)}')
    output_text = tokenizer.decode(token_ids[n_prompt_tokens:])
    print('============================================================')
    print(output_text)
    print('------------------------------------------------------------')
    print(output_text_ref)
    print('============================================================')

if __name__ == '__main__':
    pytest.main([__file__])