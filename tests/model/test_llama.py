import pytest
import torch
from transformers import LlamaConfig, LlamaTokenizer
from transformers import LlamaForCausalLM as LlamaForCausalLMRef
from dxz.model.llama import LlamaForCausalLM
from dxz.memory.kv_cache import KVCache
from dxz.model.parameters import InputParameters

@torch.inference_mode()
def test_llama2_7b_correctness():
    dtype = torch.float
    device = torch.device('cpu')
    # model
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    model_ref = LlamaForCausalLMRef.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = LlamaForCausalLM(config)
    model.load_state_dict(model_ref.state_dict())

    # input
    prompt = "Hey, are you conscious? Can you talk to me?"

    n_blocks = 100
    block_size = 16
    kv_caches = [KVCache(
        torch.zeros(size=(n_blocks, block_size, config.num_key_value_heads, config.head_dim), dtype=dtype, device=device), 
        torch.zeros(size=(n_blocks, block_size, config.num_key_value_heads, config.head_dim), dtype=dtype, device=device)) for _ in range(config.num_hidden_layers)]

    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(token_ids, dtype=torch.int, device=device)
    position_ids = torch.arange(len(token_ids), dtype=torch.int, device=device)

    input_params = InputParameters(
        num_sequences = 1, 
        q_cu_seq_lens = torch.tensor([0, len(token_ids)], dtype=torch.int ,device=device),
        kv_cu_seq_lens = torch.tensor([0, len(token_ids)], dtype=torch.int ,device=device),
        new_cache_slots = torch.arange(len(token_ids), dtype=torch.int, device=device),
        block_tables = torch.tensor([0, 1, 2 ,3 ,4 ,5 ,6 ,7], dtype=torch.int, device=device), 
        cu_blocks_lens = torch.tensor([0, 8], dtype=torch.int, device=device)
    )
    print(f'input_ids                       : {input_ids}')
    print(f'position_ids                    : {position_ids}')
    print(f'input_params.num_sequences      : {input_params.num_sequences}')
    print(f'input_params.q_cu_seq_lens      : {input_params.q_cu_seq_lens}')
    print(f'input_params.kv_cu_seq_lens     : {input_params.kv_cu_seq_lens}')
    print(f'input_params.new_cache_slots    : {input_params.new_cache_slots}')
    print(f'input_params.block_tables       : {input_params.block_tables}')
    print(f'input_params.cu_blocks_lens     : {input_params.cu_blocks_lens}')

    logits = model(input_ids, position_ids, kv_caches, input_params)
    logits_ref = model_ref(input_ids=input_ids[None, :], position_ids=position_ids[None, :])['logits'].reshape(logits.shape)

    print('logits.shape    : {logits.shape}')
    print('logits_ref.shape: {logits_ref.shape}')
    print('logits[:10]     : {logits[:10]}')
    print('logits_ref[:10] : {logits_ref[:10]}')

    assert torch.allclose(logits, logits_ref, atol=1e-2, rtol=1e-2)

