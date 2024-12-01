import os
import torch
from torch import nn, Tensor
from transformers import LlamaConfig
import safetensors.torch
from dxz.layer.rotary_embedding import RotaryEmbedding, RotaryEmbeddingRef, compute_default_inv_freq
from dxz.model.parameters import ModelParameters
from dxz.layer.attention import TorchCausalGroupedQueryPageAttention
from dxz.layer.attention import FlashCausalGroupedQueryPageAttention
from dxz._C.kernel.norm import rms_norm
from dxz._C.kernel.activation import silu

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
        self.attention = FlashCausalGroupedQueryPageAttention(
            n_qo_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim
        )
    
    def forward(self, hidden_states: Tensor, position_ids: Tensor, model_params: ModelParameters) -> Tensor:
        query = self.q_proj(hidden_states)
        key   = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = query.view(-1, self.config.num_attention_heads, self.config.head_dim)
        key   = key  .view(-1, self.config.num_key_value_heads, self.config.head_dim)
        value = value.view(-1, self.config.num_key_value_heads, self.config.head_dim)
        query, key = self.rotary_emb(query, key, position_ids)
        hidden_states = self.attention(query, key, value, model_params)
        return self.o_proj(hidden_states)

class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        down_proj = self.down_proj(silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return down_proj

class LlamaRMSNorm(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        output = torch.empty_like(hidden_states)
        rms_norm(output, hidden_states, self.weight, self.variance_epsilon)
        return output

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = LlamaSdpaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)
    
    def forward(self, hidden_states: Tensor, position_ids: Tensor, model_params: ModelParameters) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, model_params.attention_params[self.layer_id])
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
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config)
    
    def forward(self, input_ids: Tensor, position_ids: Tensor, model_params: ModelParameters) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_ids, model_params)
        return self.norm(hidden_states)

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids: Tensor, position_ids: Tensor, model_params: ModelParameters) -> Tensor:
        hidden_state = self.model(input_ids, position_ids, model_params)
        logits = self.lm_head(hidden_state)
        return logits
    
    @classmethod
    def from_safetensor(cls, model_weights_path: str, dtype: torch.dtype, device: torch.device):
        # 1. create model
        config = LlamaConfig.from_pretrained(model_weights_path)
        torch.set_default_dtype(dtype)
        with torch.device(device):
            model = cls(config)
        torch.set_default_dtype(torch.float)

        # 2. load weights
        state_dict = model.state_dict()
        loaded_set = set()
        for entry in os.scandir(model_weights_path):
            if entry.is_file() and os.path.splitext(entry.name)[1] == '.safetensors':
                print(f'load safetensor from {entry.path}')
                for name, weight in safetensors.torch.load_file(entry.path).items():
                    if name.endswith('.self_attn.rotary_emb.inv_freq'):
                        continue
                    state_dict[name].data.copy_(weight)
                    loaded_set.add(name)
        model.load_state_dict(state_dict)
        model.to(dtype)
        model.eval()

        # 3. verify
        assert len(loaded_set) == len(state_dict)

        return model
