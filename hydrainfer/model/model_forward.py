from torch import nn, Tensor
from hydrainfer.layer.rotary_embedding import RotaryEmbedding
from hydrainfer.layer.causal_attention import AttentionParameters, CausalGroupedQueryPageAttention, CausalGroupedQueryPageAttentionConfig


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
        q_proj: nn.Linear, 
        k_proj: nn.Linear, 
        v_proj: nn.Linear, 
        o_proj: nn.Linear, 
        rotary_emb: RotaryEmbedding, 
        n_qo_heads: int, 
        n_kv_heads: int, 
        head_dim: int, 
    ):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.rotary_emb = rotary_emb
        self.n_qo_heads = n_qo_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.attention = CausalGroupedQueryPageAttention(CausalGroupedQueryPageAttentionConfig(
            n_qo_heads = n_qo_heads,
            n_kv_heads = n_kv_heads,
            head_dim = head_dim, 
        ))
    def forward(self, hidden_states: Tensor, position_ids: Tensor, attention_param: AttentionParameters):
        # hidden_states (n_tokens, hidden_size)
        # position_ids (n_tokens, )
        query = self.q_proj(hidden_states) # (n_tokesn, hidden_size)
        key   = self.k_proj(hidden_states) # (n_tokesn, hidden_size)
        value = self.v_proj(hidden_states) # (n_tokesn, hidden_size)
        query = query.view(-1, self.n_qo_heads, self.head_dim) # (n_tokens, n_qo_heads, head_size)
        key   = key  .view(-1, self.n_kv_heads, self.head_dim) # (n_tokens, n_kv_heads, head_size)
        value = value.view(-1, self.n_kv_heads, self.head_dim) # (n_tokens, n_kv_heads, head_size)
        query, key = self.rotary_emb(query, key, position_ids) # query (n_tokens, n_qo_heads, head_size) key (n_tokens, n_kv_heads, head_size) note that rotary_emb is inplace operation
        hidden_states = self.attention(query, key, value, attention_param).o # (n_tokens, hidden_size)
        return self.o_proj(hidden_states) # (n_tokens, hidden_size)