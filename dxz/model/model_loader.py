import torch
from torch import nn
from transformers import PreTrainedTokenizer


def load_model_tokenizer(model_name: str, dtype: torch.dtype, device: torch.device) -> tuple[nn.Module, PreTrainedTokenizer]:
    if model_name == 'gpt2':
        from dxz.model.gpt2 import GPT2LMHeadModel
        from transformers import GPT2Tokenizer
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        n_kv_heads = model.config.n_head
        head_size = model.config.n_embd // model.config.n_head
        n_layers = model.config.n_layer
        max_seq_len  = model.config.n_positions
    elif model_name == 'meta-llama/Llama-2-7b-hf':
        from dxz.model.llama import LlamaForCausalLM
        from transformers import LlamaTokenizer
        model = LlamaForCausalLM.from_pretrained(model_name)
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        n_kv_heads = model.config.num_key_value_heads
        head_size = model.config.head_dim
        n_layers = model.config.num_hidden_layers
        max_seq_len  = model.config.max_position_embeddings
    else:
        raise Exception(f'invalid model {model_name}')
    model.to(dtype)
    model.to(device)
    model.eval()
    
    return model, tokenizer, n_kv_heads, head_size, n_layers, max_seq_len