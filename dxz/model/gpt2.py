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
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # todo
        return query

class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2Attention, self).__init__()
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.atten = Attention()
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.atten(q, k, v)
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
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
    
    def forward(self, input_ids: Tensor, position_ids: Tensor):
        input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = input_embeds + position_embeds
        for i, layer in enumerate(self.h):
            hidden_states = layer(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states
    
    def load_weights(self, state_dict_ref: dict[str, Tensor]):
        for name, weight in self.state_dict().items():
            if any(needle in name for needle in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
                weight.data.copy_(state_dict_ref[name].t())
            else:
                weight.data.copy_(state_dict_ref[name])

def test_shape():
    def count_parameters(model: nn.Module):
        total = 0
        param_required_grad = 0
        for param in model.parameters():
            total += param.numel()
            if param.requires_grad:
                param_required_grad += param.numel()
        return total, param_required_grad

    config = GPT2Config()
    model = GPT2Model(config)

    count = count_parameters(model)
    print(count)
    assert count[1] == 124439808

    print('pass')

def test_model_name():
    def parameters_list(model: nn.Module):
        param_list = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_list.append((name, f"{param.shape}"))
        return param_list

    config = GPT2Config()
    model = GPT2Model(config)
    l = parameters_list(model)
    print(len(l))

    from transformers import GPT2Model as GPT2ModelRef
    model_ref = GPT2ModelRef.from_pretrained('gpt2')
    l_ref = parameters_list(model_ref)
    print(len(l_ref))
    for (name, param), (name_ref, param_ref) in zip(l, l_ref):
        print(name, param)
        print(name_ref, param_ref)
        assert name == name_ref
        # assert param == param_ref
        if param == param_ref:
            print('--------------------------------')
        else:
            print('------------- dif --------------')
    print('pass')

def test_model_output():
    config = GPT2Config()
    model = GPT2Model(config)
    from transformers import GPT2Model as GPT2ModelRef
    model_ref = GPT2ModelRef.from_pretrained('gpt2')

    state_dict = model_ref.state_dict()
    model.load_weights(state_dict)

    input_ids = torch.arange(10)
    position_ids = torch.arange(10)

    output = model(input_ids=input_ids, position_ids=position_ids)
    output_ref = model_ref(input_ids=input_ids, position_ids=position_ids)

    assert torch.allclose(output, output_ref['last_hidden_state'], rtol=1e-3, atol=1e-3)
    print('pass')

if __name__ == '__main__':
    test_model_output()