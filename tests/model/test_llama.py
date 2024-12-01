import pytest
import torch
from transformers import LlamaConfig, LlamaTokenizer
from transformers import LlamaForCausalLM as LlamaForCausalLMRef
from dxz.model.llama import LlamaForCausalLM
from dxz.memory.kv_cache import KVCache
from dxz.model.parameters import AttentionParameters, ModelParameters

@torch.inference_mode()
def test_llama2_7b_correctness():
    dtype = torch.half
    device = torch.device('cuda:0')
    # model
    pretrained_model_name_or_path = 'meta-llama/Llama-2-7b-hf'
    tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path)
    config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)
    model_ref = LlamaForCausalLMRef.from_pretrained(pretrained_model_name_or_path)
    model = LlamaForCausalLM(config)
    model.load_state_dict(model_ref.state_dict())
    model.to(dtype)
    model.to(device)

    # input
    prompt = "Hey, are you conscious? Can you talk to me?"

    n_blocks = 100
    block_size = 16
    kv_caches = [KVCache(n_blocks, block_size, config.num_key_value_heads, config.head_dim, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)]

    token_ids = tokenizer.encode(prompt)
    input_ids_ref = torch.tensor(token_ids, dtype=torch.int)
    position_ids_ref = torch.arange(len(token_ids), dtype=torch.int)
    input_ids = torch.tensor(token_ids, dtype=torch.int, device=device)
    position_ids = torch.arange(len(token_ids), dtype=torch.int, device=device)
    model_params = ModelParameters([AttentionParameters(
        num_sequences = 1, 
        q_cu_seq_lens = torch.tensor([0, len(token_ids)], dtype=torch.int ,device=device),
        kv_cu_seq_lens = torch.tensor([0, len(token_ids)], dtype=torch.int ,device=device),
        new_cache_slots = torch.arange(len(token_ids), dtype=torch.int, device=device),
        block_tables = torch.tensor([0, 1, 2 ,3 ,4 ,5 ,6 ,7], dtype=torch.int, device=device), 
        cu_blocks_lens = torch.tensor([0, 8], dtype=torch.int, device=device), 
        kv_cache = kv_caches[layer_id], 
        all_sequences_decode = False,
        q_max_seq_len = 128, 
        kv_max_seq_len = 128, 
        ) for layer_id in range(config.num_hidden_layers)])

    logits = model(input_ids, position_ids, model_params)
    logits_ref = model_ref(input_ids=input_ids_ref[None, :], position_ids=position_ids_ref[None, :])['logits'].reshape(logits.shape)

    print(f'logits.shape    : {logits.shape}')
    print(f'logits_ref.shape: {logits_ref.shape}')
    print(f'logits[:10]     : {logits[0, :10]}')
    print(f'logits_ref[:10] : {logits_ref[0, :10]}')

    logits = logits.to(logits_ref)
    assert torch.allclose(logits, logits_ref, atol=0.2, rtol=0.2)

@torch.inference_mode()
def test_llama3_8b_instruct_correctness():
    dtype = torch.half
    device = torch.device('cuda:0')
    # model
    pretrained_model_name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)
    model_ref = LlamaForCausalLMRef.from_pretrained(pretrained_model_name_or_path)
    model = LlamaForCausalLM(config)
    model.load_state_dict(model_ref.state_dict())
    model.to(dtype)
    model.to(device)


    # input

    n_blocks = 100
    block_size = 16
    kv_caches = [KVCache(n_blocks, block_size, config.num_key_value_heads, config.head_dim, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)]

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )
    input_ids_ref = torch.tensor(token_ids, dtype=torch.int)
    position_ids_ref = torch.arange(len(token_ids), dtype=torch.int)
    input_ids = torch.tensor(token_ids, dtype=torch.int, device=device)
    position_ids = torch.arange(len(token_ids), dtype=torch.int, device=device)
    model_params = ModelParameters([AttentionParameters(
        num_sequences = 1, 
        q_cu_seq_lens = torch.tensor([0, len(token_ids)], dtype=torch.int ,device=device),
        kv_cu_seq_lens = torch.tensor([0, len(token_ids)], dtype=torch.int ,device=device),
        new_cache_slots = torch.arange(len(token_ids), dtype=torch.int, device=device),
        block_tables = torch.tensor([0, 1, 2 ,3 ,4 ,5 ,6 ,7], dtype=torch.int, device=device), 
        cu_blocks_lens = torch.tensor([0, 8], dtype=torch.int, device=device), 
        kv_cache = kv_caches[layer_id], 
        all_sequences_decode = False,
        q_max_seq_len = 128, 
        kv_max_seq_len = 128, 
    ) for layer_id in range(config.num_hidden_layers)])

    logits = model(input_ids, position_ids, model_params)
    logits_ref = model_ref(input_ids=input_ids_ref[None, :], position_ids=position_ids_ref[None, :])['logits'].reshape(logits.shape)

    print(f'logits.shape    : {logits.shape}')
    print(f'logits_ref.shape: {logits_ref.shape}')
    print(f'logits[:10]     : {logits[0, :10]}')
    print(f'logits_ref[:10] : {logits_ref[0, :10]}')

    logits = logits.to(logits_ref)
    assert torch.allclose(logits, logits_ref, atol=0.2, rtol=0.2)

if __name__ == '__main__':
    pytest.main([__file__])
    # test_llama2_7b_correctness()
    # test_llama3_8b_instruct_correctness()