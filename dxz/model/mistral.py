import torch
from torch import nn, Tensor
from transformers import MistralConfig, MistralForCausalLM

class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class MistralMLP(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class MistralAttention(nn.Module):
    def __init__(self, config: MistralConfig):
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(hidden_state: Tensor, position_ids: Tensor) -> Tensor:
        q = self.q_proj(hidden_state).view(-1, self.config.num_attention_heads, self.config.head_dim)
        k = self.k_proj(hidden_state).view(-1, self.config.num_key_value_heads, self.config.head_dim)
        v = self.v_proj(hidden_state).view(-1, self.config.num_key_value_heads, self.config.head_dim)
        
        # attention kernel 
        sm_scale = 1. / math.sqrt(self.config.head_dim)
        scores = torch.einsum("qhd,khd->hqk", query.to(torch.float32), key.to(torch.float32))
        scores *= sm_scale

        # todo mask
        # todo slide window
        if mask == None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        scores = torch.softmax(scores, dim=-1)

        h = torch.einsum("hqk,khd->qhd", scores, value)
        
        return self.o_proj(h)

class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MistralAttention(config)
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(hidden_states: Tensor, position_ids: Tensor) -> Tensor:
        h = hidden_states + self.self_attn(self.input_layernorm(hidden_states), position_ids)
        return h + self.mlp(self.post_attention_layernorm(h))

class MistralModel(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size=config.vocab_size,num_embeddings=config.hidden_size)
        self.layers = [MistralDecoderLayer(config) for _ in config.num_hidden_layers]
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(input_ids: Tensor, position_ids: Tensor) -> Tensor:
        h = self.embed_tokens(input_ids)
        for _, layer in enumerate(self.layers):
            h = layer(h)
        return self.norm(h)

class MistralForCausalLM(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.model = MistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(input_ids: Tensor, position_ids: Tensor) -> Tensor:
        h = self.model(input_ids, position_ids)
        self.lm_head(h)