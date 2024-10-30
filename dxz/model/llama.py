import torch
from torch import nn, Tensor
from transformers import LlamaConfig
from transformers import LlamaForCausalLM as LlamaForCausalLMRef
from dxz.layer.rotary_embedding import RotaryEmbedding, compute_default_inv_freq
from dxz.model.parameters import InputParameters
from dxz.memory.kv_cache import KVCache
from dxz.layer.attention import Attention
from dxz.layer.attention import FlashAttention

class LlamaSdpaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            rotary_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            inv_freq=compute_default_inv_freq(
                rotary_dim=config.head_dim,
                theta=config.rope_theta
                ),
            interleaved=False
            )
        if self.config.torch_dtype == 'float16':
            self.attention = FlashAttention(
                n_qo_heads=config.num_attention_heads,
                n_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim
            )
        else:
            self.attention = Attention(
                n_qo_heads=config.num_attention_heads,
                n_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim
                )
    
    def forward(self, hidden_states: Tensor, position_ids: Tensor, kv_cache: KVCache, input_params: InputParameters) -> Tensor:
        query = self.q_proj(hidden_states)
        key   = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = query.view(-1, self.config.num_attention_heads, self.config.head_dim)
        key   = key  .view(-1, self.config.num_key_value_heads, self.config.head_dim)
        value = value.view(-1, self.config.num_key_value_heads, self.config.head_dim)
        query, key = self.rotary_emb(query, key, position_ids)
        hidden_states = self.attention(query, key, value, kv_cache, input_params)
        return self.o_proj(hidden_states)

class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU() 
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return down_proj

class LlamaRMSNorm(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaSdpaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)
    
    def forward(self, hidden_states: Tensor, position_ids: Tensor, kv_cache: KVCache, input_params: InputParameters) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, kv_cache, input_params)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config)
    
    def forward(self, input_ids: Tensor, position_ids: Tensor, kv_caches: list[KVCache], input_params: InputParameters) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_ids, kv_caches[i], input_params)
        return self.norm(hidden_states)

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids: Tensor, position_ids: Tensor, kv_caches: list[KVCache], input_params: InputParameters) -> Tensor:
        hidden_state = self.model(input_ids, position_ids, kv_caches, input_params)
        logits = self.lm_head(hidden_state)
        return logits
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        assert pretrained_model_name_or_path == "meta-llama/Llama-2-7b-hf", f'{pretrained_model_name_or_path} is not support'
        model_ref = LlamaForCausalLMRef.from_pretrained(pretrained_model_name_or_path)
        model = cls(model_ref.config)
        state_dict_ref = model_ref.state_dict()
        for name, weight in model.state_dict().items():
            weight.data.copy_(state_dict_ref[name])
        return model
