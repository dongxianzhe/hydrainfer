import torch
from torch import nn, Tensor
from transformers import MixtralConfig
from torch.nn import functional
from typing import Optional
from dxz.memory.kv_cache import KVCache
from dxz.model.parameters import InputParameters
from dxz.layer.rotary_embedding import compute_default_inv_freq, RotaryEmbedding

class MixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class MixtralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, hidden_states: Tensor, streams: Optional[list[torch.cuda.Stream]]=None) -> Tensor:
        if streams:
            with torch.cuda.stream(streams[0]):
                h1 = self.act_fn(self.w1(hidden_states))
            with torch.cuda.stream(streams[1]):
                h2 = self.w3(hidden_states)
            current_hidden_states =  h1 * h2
            current_hidden_states = self.w2(current_hidden_states)
            return current_hidden_states
        else:
            current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
            current_hidden_states = self.w2(current_hidden_states)
            return current_hidden_states


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts)
        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(config.num_local_experts)])
    
    def forward(self, hidden_states: Tensor, streams: Optional[list[torch.cuda.Stream]]=None) -> Tensor:
        # hidden_states (n_tokens, hidden_state)
        router_logits = self.gate(hidden_states) # (n_tokens, num_local_experts)
        routing_weights = functional.softmax(router_logits, dim=-1, dtype=torch.float) # (n_tokens, num_local_experts)
        routing_weights, selected_experts = torch.topk(routing_weights, self.config.num_experts_per_tok, dim=-1)
        routing_weights = routing_weights.to(hidden_states.dtype)
        # routing_weights(n_tokens, num_experts_per_tok) selected_experts(n_tokens, num_experts_per_tok)
        expert_mask = functional.one_hot(selected_experts, num_classes=self.config.num_local_experts).permute(2, 1, 0)
        # expert_mask (n_tokens, num_experts_per_tok, num_local_experts) 
        # -- permute -> (num_local_experts, num_experts_per_tok, n_tokens)  

        final_hidden_states = torch.zeros(size=(hidden_states.shape[0], self.config.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        for expert_idx in range(self.config.num_local_experts):
            if streams:
                with torch.cuda.stream(streams[expert_idx % 2]):
                    # exper_mask[expert_idx] (num_experts_per_tok, n_tokens)
                    weight_ids, token_ids = torch.where(expert_mask[expert_idx])
                    current_state = hidden_states[token_ids]
                    # current_state (n, hidden_size)
                    current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[token_ids, weight_ids, None]
                    # current_hidden_states (n, hidden_size)
                    final_hidden_states.index_add_(dim=0, index=token_ids, source=current_hidden_states.to(hidden_states.dtype))
            else:
                # exper_mask[expert_idx] (num_experts_per_tok, n_tokens)
                weight_ids, token_ids = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_ids]
                # current_state (n, hidden_size)
                current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[token_ids, weight_ids, None]
                # current_hidden_states (n, hidden_size)
                final_hidden_states.index_add_(dim=0, index=token_ids, source=current_hidden_states.to(hidden_states.dtype))

        return final_hidden_states

class MixtralAttention(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            rotary_dim = self.head_dim, 
            max_position_embeddings = self.max_position_embeddings, 
            inv_freq = compute_default_inv_freq(rotary_dim=self.head_dim, theta=self.rope_theta), 
            interleaved = False, 
        )
        self.rotary_emb = MixtralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    
    def forward(h: Tensor, kv_cache: KVCache, input_params: InputParameters) -> Tensor:
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        # todo rotary embedding
        h = self.attention(q, k, v, kv_cache, input_params)
        o = self.o_proj(h)
        return o

class MixtralDecoderLayer(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.atten = atten
        # todo
        self.self_attn = MixtralAttention(config)
        self.block_sparse_moe = MixtralSparseMoeBlock(config)
        self.input_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(h: Tensor, kv_cache: KVCache, input_params: InputParameters) -> Tensor: 
        r = h
        h = self.input_layernorm(h)
        h = self.self_attn(h, kv_cache, input_params)
        h = r + h

        r = h
        h = self.post_attention_layernorm(h)
        h = self.block_sparse_moe(h)
        h = r + h
        return h

class MixtralModel(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MixtralDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(input_ids: Tensor, position_ids: Tensor, kv_caches: list[KVCache], input_params: InputParameters) -> dict[str, Tensor]:
        h = self.embed_tokens(input_ids)
        all_hidden_states = []
        for i, layer in enumerate(self.layers):
            all_hidden_states.append(h)
            h = self.layer(h, kv_caches[i], input_params)
        all_hidden_states.append(h)
        h = self.norm(h)
        return {
            "last_hidden_state" : h,
            "hidden_states" : all_hidden_states, # the input of [layer1, layer2, ... layer12, lm_heads]
        }

    

class MixtralForCausalLM(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.model = MixtralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(input_ids: Tensor, position_ids: Tensor, kv_caches: list[KVCache], input_params: InputParameters) -> dict[str, Tensor]:
        h = self.model(input_ids, position_ids, kv_caches, input_params)['last_hidden_state']
        lm_logits = self.lm_head(h)
        return {"logits" : lm_logits}


if __name__ == '__main__':
    config = MixtralConfig()
    # print(config._attn_implementation)
    # print(config._attn_implementation)
    # model = MixtralSparseMoeBlock()
