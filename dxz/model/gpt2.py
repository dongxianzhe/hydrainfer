from typing import List, Optional

import math
import torch
from torch import nn
from torch import Tensor
from transformers import GPT2Config
from dxz.model.parameters import InputParameters
from dxz.layer.activation import NewGELUActivation
from dxz.memory.kv_cache import KVCache
from dxz.layer.attention import FlashCausalGroupedQueryPageAttention

class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2Attention, self).__init__()
        self.config = config
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.atten = FlashCausalGroupedQueryPageAttention(
            n_qo_heads=config.n_head,
            n_kv_heads=config.n_head,
            head_dim=config.hidden_size // config.n_head
        )
    
    def forward(self, hidden_states: Tensor, kv_cache: KVCache, input_params: InputParameters) -> Tensor:
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.atten(q, k, v, kv_cache, input_params)
        attn_output = self.c_proj(attn_output)
        return attn_output

class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2MLP, self).__init__()
        assert config.n_inner is None, "only support intermediate = 4 * hidden_size now"
        intermediate_size = config.hidden_size * 4
        self.c_fc = nn.Linear(config.hidden_size, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, config.hidden_size, bias=True)
        self.act = NewGELUActivation()
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config=config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden_states: Tensor, kv_cache: KVCache, input_params: InputParameters) -> Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, kv_cache, input_params)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states

class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2Model, self).__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
    
    def forward(self, input_ids: Tensor, position_ids: Tensor, kv_caches: list[KVCache], input_params: InputParameters) -> dict[str, Tensor]:
        input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = input_embeds + position_embeds

        for i, layer in enumerate(self.h):
            hidden_states = layer(hidden_states, kv_caches[i], input_params)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states
    
class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2LMHeadModel, self).__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
    
    def forward(self, input_ids: Tensor, position_ids: Tensor, kv_caches: list[KVCache], input_params: InputParameters) -> dict[str, Tensor]:
        hidden_state = self.transformer(input_ids, position_ids, kv_caches, input_params)
        lm_logits = self.lm_head(hidden_state)
        return lm_logits
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        assert pretrained_model_name_or_path == 'gpt2', f'{pretrained_model_name_or_path} is not support'
        from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
        model_ref = GPT2LMHeadModelRef.from_pretrained(pretrained_model_name_or_path)
        model = cls(model_ref.config)
        state_dict_ref = model_ref.state_dict()
        for name, weight in model.state_dict().items():
            if any(needle in name for needle in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
                weight.data.copy_(state_dict_ref[name].t())
            else:
                weight.data.copy_(state_dict_ref[name])
        return model