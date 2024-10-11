from typing import List, Optional

import math
import torch
from torch import nn
from torch import Tensor
from transformers import GPT2Config
from dxz.model.parameters import InputParameters
from dxz.layer.activation import NewGELUActivation
from dxz.memory.kv_cache import KVCache
from dxz.layer.attention import Attention

# class Attention(nn.Module):
#     def __init__(self, n_qo_heads: int, n_kv_heads: int, head_dim: int):
#         super().__init__()
#         self.n_qo_heads  = n_qo_heads
#         self.n_kv_heads = n_kv_heads
#         self.head_dim   = head_dim

#     def forward(self, query: Tensor, key: Tensor, value: Tensor, kv_cache: tuple[Tensor, Tensor], input_params: InputParameters) -> Tensor:
#         # query (q_seq_len, num_heads * head_size)
#         # key   (k_seq_len, num_heads * head_size)
#         # value (k_seq_len, num_heads * head_size)
#         # only support q_seq_len == k_seq_len because not support flat tensor and kv cache. the batch can only be one now
#         q = query.view(-1, self.n_qo_heads, self.head_dim)
#         k = key.view(-1, self.n_kv_heads, self.head_dim)
#         v = value.view(-1, self.n_kv_heads, self.head_dim)
#         # set kv cache
#         n_old_token = input_params.cache_length
#         n_new_token = q.shape[0]
#         n_token = n_old_token + n_new_token
#         k_cache, v_cache = kv_cache
#         k_cache[n_old_token : n_token].data.copy_(k)
#         v_cache[n_old_token : n_token].data.copy_(v)

#         # creat sm_scale
#         sm_scale = 1. / math.sqrt(q.shape[-1])

#         # create mask
#         q_seq_len = n_new_token
#         kv_seq_len = n_token
#         if q_seq_len == kv_seq_len: # prefill state
#             mask = torch.tril(torch.ones(size=(q_seq_len, q_seq_len), dtype=torch.bool, device=q.device))
#         else: # decode state
#             mask = None
#         k = k_cache[: kv_seq_len]
#         v = v_cache[: kv_seq_len]

#         # q @ k
#         scores = torch.einsum("qhd,khd->hqk", q.to(torch.float32), k.to(torch.float32))

#         # sm_scale
#         if sm_scale is not None:
#             scores *= sm_scale
#         # casual mask        
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))

#         # softmax
#         scores = torch.softmax(scores, dim=-1)

#         # s@v
#         o = torch.einsum("hqk,khd->qhd", scores, v)
#         return o.view(q_seq_len, -1)
#         # attn_output = attn_output.reshape(q_seq_len, self.config.hidden_size)

class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2Attention, self).__init__()
        self.config = config
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.atten = Attention(
            n_qo_heads=config.n_head,
            n_kv_heads=config.n_head,
            head_dim=config.hidden_size // config.n_head
        )
    
    def forward(self, hidden_states: Tensor, kv_cache: KVCache, input_params: InputParameters) -> Tensor:
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.atten(q, k, v, kv_cache, input_params)
        # print(f'attn_output.shape {attn_output.shape}')
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

        all_hidden_states = ()
        for i, layer in enumerate(self.h):
            all_hidden_states += (hidden_states, )
            hidden_states = layer(hidden_states, kv_caches[i], input_params)
        hidden_states = self.ln_f(hidden_states)
        all_hidden_states += (hidden_states, )
        return {
            "last_hidden_state" : hidden_states,
            "hidden_states" : all_hidden_states, # the input of [layer1, layer2, ... layer12, lm_heads]
        }
    
    def load_state_dict(self, state_dict_ref: dict[str, Tensor]):
        for name, weight in self.state_dict().items():
            if any(needle in name for needle in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
                weight.data.copy_(state_dict_ref[name].t())
            else:
                weight.data.copy_(state_dict_ref[name])

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids: Tensor, position_ids: Tensor, kv_caches: list[KVCache], input_params: InputParameters) -> dict[str, Tensor]:
        hidden_state = self.transformer(input_ids, position_ids, kv_caches, input_params)['last_hidden_state']
        lm_logits = self.lm_head(hidden_state)
        return {'logits' : lm_logits}
    
    def load_state_dict(self, state_dict_ref: dict[str, Tensor]):
        for name, weight in self.state_dict().items():
            if 'lm_head' in name:
                weight.data.copy_(state_dict_ref['transformer.wte.weight']) 
            else:
                if any(needle in name for needle in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
                    weight.data.copy_(state_dict_ref[name].t())
                else:
                    weight.data.copy_(state_dict_ref[name])
