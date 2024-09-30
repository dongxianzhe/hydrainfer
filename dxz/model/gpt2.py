from typing import List, Optional

import math
import torch
from torch import nn
from torch import Tensor
from transformers import GPT2Config

class NewGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class Attention(nn.Module):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, sm_scale: Optional[float] = None, logits_soft_cap: Optional[float] = None, alibi_biases: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        # query (q_seq_len, num_heads, head_size)
        # key   (k_seq_len, num_heads, head_size)
        # value (k_seq_len, num_heads, head_size)
        # alibi_biases (num_heads, q_seq_len, k_seq_len)
        # only support q_seq_len == k_seq_len because not support flat tensor and kv cache. the batch can only be one now
        scores = torch.einsum("qhd,khd->hqk", query.to(torch.float32), key.to(torch.float32))
        if sm_scale is not None:
            scores *= sm_scale

        if logits_soft_cap is not None:
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        
        if alibi_biases is not None:
            scores += alibi_biases
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        scores = torch.softmax(scores, dim=-1)
        return torch.einsum("hqk,khd->qhd", scores, value)

class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2Attention, self).__init__()
        self.config = config
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.atten = Attention()
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        q_seq_len = hidden_states.shape[0]

        qkv = self.c_attn(hidden_states)

        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q = q.view(-1, self.config.n_head, self.config.hidden_size // self.config.n_head)
        k = k.view(-1, self.config.n_head, self.config.hidden_size // self.config.n_head)
        v = v.view(-1, self.config.n_head, self.config.hidden_size // self.config.n_head)

        mask = torch.tril(torch.ones(size=(q_seq_len, q_seq_len), dtype=torch.bool))

        attn_output = self.atten(q, k, v, sm_scale = 1. / math.sqrt(q.shape[-1]), mask=mask)
        attn_output = attn_output.reshape(q_seq_len, self.config.hidden_size)

        attn_output = self.c_proj(attn_output)

        # print_once(f'attn_output {attn_output.shape} {attn_output}')

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

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
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
    
    def forward(self, input_ids: Tensor, position_ids: Tensor):
        input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = input_embeds + position_embeds

        all_hidden_states = ()
        for _, layer in enumerate(self.h):
            all_hidden_states += (hidden_states, )
            hidden_states = layer(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        all_hidden_states += (hidden_states, )
        return {
            "last_hidden_state" : hidden_states,
            "hidden_states" : all_hidden_states,
        }
    
    def load_weights(self, state_dict_ref: dict[str, Tensor]):
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
    
    def forward(self, input_ids: Tensor, position_ids: Tensor) -> Tensor:
        hidden_state = self.transformer(input_ids, position_ids)['last_hidden_state']
        lm_logits = self.lm_head(hidden_state)
        return {'logits' : lm_logits}
    
    def load_weights(self, state_dict_ref: dict[str, Tensor]):
        for name, weight in self.state_dict().items():
            if 'lm_head' in name:
                weight.data.copy_(state_dict_ref['transformer.wte.weight']) 
            else:
                if any(needle in name for needle in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
                    weight.data.copy_(state_dict_ref[name].t())
                else:
                    weight.data.copy_(state_dict_ref[name])
