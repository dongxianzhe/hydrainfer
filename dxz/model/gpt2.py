from typing import List, Optional

import math
import torch
from torch import nn
from torch import Tensor
from transformers import GPT2Config

def print_once(message):
    if not hasattr(print_once, 'has_printed'):
        print_once.has_printed = False  # 初始化属性
    if not print_once.has_printed:
        print(message)
        print_once.has_printed = True  # 更新状态

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
        # return hidden_states
    
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
    model.eval()
    model_ref.eval()


    state_dict = model_ref.state_dict()
    model.load_weights(state_dict)

    # check param weights equal
    s1 = model.state_dict()
    s2 = model_ref.state_dict()
    for name, weight in s1.items():
        if any(needle in name for needle in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
            weight_ref = s2[name].t()
        else:
            weight_ref = s2[name]

        assert torch.allclose(weight, weight_ref, rtol=1e-3, atol=1e-3), f"layer {name} not equal"
        
    input_ids = torch.arange(20)
    position_ids = torch.arange(20)

    output = model(input_ids=input_ids, position_ids=position_ids)
    output_ref = model_ref(input_ids=input_ids, position_ids=position_ids, output_hidden_states = True)
    for i, (layer_i, layer_i_ref) in enumerate(zip(output['hidden_states'], output_ref['hidden_states'])):
        print(f'---------------layer {i}---------------')
        print(f'layer_i {layer_i.shape} layer_i_ref {layer_i_ref.shape}')
        layer_i = layer_i.reshape(-1)
        layer_i_ref = layer_i_ref.reshape(-1)
        print(f'layer_i {layer_i.shape} layer_i_ref {layer_i_ref.shape}')
        print(layer_i)
        print(layer_i_ref)
        print(torch.allclose(layer_i.reshape(-1, 768)[0], layer_i_ref.reshape(-1, 768)[0], rtol=1e-3, atol=1e-3))
        print(torch.allclose(layer_i, layer_i_ref, rtol=1e-3, atol=1e-3))

    print(f"{len(output_ref['hidden_states'])}")

    print(output['last_hidden_state'].view(-1)[0])
    print(output_ref['last_hidden_state'].view(-1)[0])
    
    assert torch.allclose(output['last_hidden_state'], output_ref['last_hidden_state'], rtol=1e-3, atol=1e-3), "outout not equal"
    print('pass')

if __name__ == '__main__':
    test_model_output()
    # x = torch.arange(12).reshape(2, 6)
    # y = x.chunk(chunks=3, dim=-1)
    # print(x)
    # print(y)