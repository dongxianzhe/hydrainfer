from torch import nn, Tensor
from typing import Optional
from hydrainfer.layer.rotary_embedding import RotaryEmbedding, compute_default_inv_freq
from hydrainfer.layer.causal_attention import AttentionParameters, CausalGroupedQueryPageAttention, CausalGroupedQueryPageAttentionConfig
from hydrainfer.model.parameters import LanguageModelParameters


class UpDownMLP:
    def __init__(
        self, 
        up_proj: nn.Linear, 
        down_proj: nn.Linear, 
        activation: nn.Module
    ):
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.activation = activation

    def forward(self, h: Tensor) -> Tensor:
        return self.down_proj(self.activation(self.up_proj(h)))


class GateUpDownMLP:
    def __init__(
        self, 
        gate_proj: nn.Linear, 
        up_proj: nn.Linear, 
        down_proj: nn.Linear, 
        activation: nn.Module
    ):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.activation = activation
    def forward(self, h: Tensor) -> Tensor:
        return self.down_proj(self.activation(self.gate_proj(h)) * self.up_proj(h))


class ROPECausalGroupedQueryPageAttention:
    def __init__(
        self, 
        n_qo_heads: int, 
        n_kv_heads: int, 
        head_dim: int, 
        rotary_emb: Optional[RotaryEmbedding] = None, 
        q_proj: Optional[nn.Linear] = None, 
        k_proj: Optional[nn.Linear] = None, 
        v_proj: Optional[nn.Linear] = None, 
        qkv_proj: Optional[nn.Linear] = None, 
        o_proj: Optional[nn.Linear] = None, 
    ):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.qkv_proj = qkv_proj
        self.rotary_emb = rotary_emb
        self.n_qo_heads = n_qo_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.attention = CausalGroupedQueryPageAttention(CausalGroupedQueryPageAttentionConfig(
            n_qo_heads = n_qo_heads,
            n_kv_heads = n_kv_heads,
            head_dim = head_dim, 
        ))
    def forward(self, hidden_states: Tensor, position_ids: Tensor, attention_param: AttentionParameters) -> Tensor:
        # hidden_states (n_tokens, hidden_size)
        # position_ids (n_tokens, )
        if self.qkv_proj is not None:
            qkv = self.qkv_proj(hidden_states)
            query = qkv[:, :self.n_qo_heads * self.head_dim]
            key = qkv[:, self.n_qo_heads * self.head_dim:self.n_qo_heads * self.head_dim + self.n_kv_heads * self.head_dim]
            value = qkv[:, self.n_qo_heads * self.head_dim + self.n_kv_heads * self.head_dim:]
        else:
            query = self.q_proj(hidden_states) # (n_tokesn, hidden_size)
            key   = self.k_proj(hidden_states) # (n_tokesn, hidden_size)
            value = self.v_proj(hidden_states) # (n_tokesn, hidden_size)
        query = query.view(-1, self.n_qo_heads, self.head_dim) # (n_tokens, n_qo_heads, head_size)
        key   = key  .view(-1, self.n_kv_heads, self.head_dim) # (n_tokens, n_kv_heads, head_size)
        value = value.view(-1, self.n_kv_heads, self.head_dim) # (n_tokens, n_kv_heads, head_size)
        if self.rotary_emb is not None:
            query, key = self.rotary_emb(query, key, position_ids) # query (n_tokens, n_qo_heads, head_size) key (n_tokens, n_kv_heads, head_size) note that rotary_emb is inplace operation
        hidden_states = self.attention(query, key, value, attention_param).o # (n_tokens, hidden_size)
        if self.o_proj is not None:
            hidden_states = self.o_proj(hidden_states) # (n_tokens, hidden_size)
        return hidden_states


class DecoderLayer:
    def __init__(self, attention: nn.Module, mlp: nn.Module, norm_1: nn.Module, norm_2: nn.Module, layer_id: int, n_layers: int):
        self.attention = attention
        self.mlp = mlp
        self.norm_1 = norm_1
        self.norm_2 = norm_2
        self.layer_id = layer_id
        self.n_layers = n_layers

    def forward(self, hidden_states: Tensor, position_ids: Tensor, model_params: LanguageModelParameters) -> Tensor:
        hidden_states = hidden_states + self.attention(self.norm_1(hidden_states), position_ids, model_params.attention_params[self.layer_id]) # (n_tokens, hidden_size)

        # if it is last layer we discared tokens which is not sampled to reduce redundent computation in the last ffn layer
        if not model_params.all_sequences_decode and self.layer_id == self.n_layers - 1:
            hidden_states = hidden_states[model_params.selected_token_ids, :]

        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        return hidden_states