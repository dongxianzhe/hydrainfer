import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from torch import Tensor
import torch.nn.functional as F
from hydrainfer.layer.norm import RMSNorm
from hydrainfer.model.parameters import LanguageModelParameters, LanguageModelOutput
from hydrainfer.layer.causal_attention import AttentionParameters
from hydrainfer.layer.moe import MoEGate, MoeGateConfig

@dataclass
class RopeScaling:
    beta_fast: int
    beta_slow: int
    factor: int
    mscale: float
    mscale_all_dim: float
    original_max_position_embeddings: int
    type: str

@dataclass
class DeepSeekV2Config:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    n_routed_experts: int
    first_k_dense_replace: int
    moe_layer_freq: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    num_attention_heads: int
    kv_lora_rank: int
    v_head_dim: int
    attention_bias: bool
    max_position_embeddings: int
    rope_scaling: RopeScaling
    q_lora_rank: Optional[int]
    intermediate_size: int
    scoring_func: str
    n_routed_experts: int
    topk_method: str
    num_experts_per_tok: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    moe_intermediate_size: int
    n_shared_experts: int

class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding):
    def __init__(self, config: DeepSeekV2Config):
        assert config.rope_scaling is not None
        assert config.rope_scaling.type == "yarn"
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )



class DeepseekV2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, h: Tensor) -> Tensor:
        # (n_tokens, hidden_size)
        return self.down_proj(self.act_fn(self.gate_proj(h)) * self.up_proj(h)) # (n_tokens, hidden_size)



class DeepseekV2MoE(nn.Module):
    def __init__(self, config: DeepSeekV2Config):
        super().__init__()
        assert self.config.scoring_func == "softmax", f"scoring_func {self.config.scoring_func} is not supported"
        assert self.config.topk_method == "greedy", f"topk method {self.config.topk_method} is not supported"
        assert self.config.norm_topk_prob == False, "norm topk prob is not supported"
        assert self.config.routed_scaling_factor == 1.0, "routed_scaling_factor is not supported"
        self.experts = nn.ModuleList([
            DeepseekV2MLP(config.hidden_size, config.moe_intermediate_size) for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(MoeGateConfig(
            hidden_size = config.hidden_size, 
            n_routed_experts = config.n_routed_experts, 
            num_experts_per_tok = config.num_experts_per_tok, 
        ))
        self.shared_experts = DeepseekV2MLP(config.hidden_size, intermediate_size=config.n_shared_experts * config.moe_intermediate_size)

    def forward(self, h: Tensor) -> Tensor:
        # h (n_tokens, hidden_size)
        topk_weight, topk_idx = self.gate(h) # (n_tokens, n_eperts_per_tok) (n_tokens, n_eperts_per_tok) 
        



class CompressedKVCache:
    def __init__(self, cache: Tensor) -> None:
        assert cache.dim() == 4, f"cache shape {cache.shape} should be (n_blocks, block_size, n_heads=1, n_kv_lora_rank + n_qk_nope_head_dim)"
        self.cache = cache
    
    def set_kv_cache(self, compressed_kv: Tensor, k_pe: Tensor, slot_ids: Tensor) -> None:
        # compressed_kv (n_tokens, n_heads=1, kv_lora_rank) 
        # k_pe (n_tokens, n_heads=1, qk_rope_head_dim)
        # slot_ids (n_tokens, )
        assert compressed_kv.shape[1] == 1, f'compressed_kv only should have only one head, but got {compressed_kv.shape[1]}'
        assert k_pe.shape[1] == 1, f'compressed_kv only should have only one head, but got {k_pe.shape}'
        assert slot_ids.shape[0] == compressed_kv.shape[0], f"{slot_ids.shape} {compressed_kv.shape}"
        assert slot_ids.shape[0] == k_pe.shape[0], f"{slot_ids.shape} {k_pe.shape}"
        assert slot_ids.device == compressed_kv.device, f"{slot_ids.device} {compressed_kv.device}"
        assert slot_ids.device == k_pe.device, f"{slot_ids.device} {k_pe.device}"

        num_tokens: int = slot_ids.shape[0]
        kv_lora_rank: int = compressed_kv.shape[1]
        qk_rope_head_dim: int = k_pe.shape[1]
        for i in range(num_tokens):
            block_id = slot_ids[i] // self.block_size
            block_offset = slot_ids[i] % self.block_size
            self.cache[block_id, block_offset, :, :kv_lora_rank] = compressed_kv[i, :, :]
            self.cache[block_id, block_offset, :, kv_lora_rank:] = k_pe[i, :, :]

class DeepseekV2Attention(nn.Module):
    def __init__(self, config: DeepSeekV2Config):
        super().__init__()
        self.config = config
        assert config.q_lora_rank is None
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim), bias=config.attention_bias)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=config.attention_bias)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim), bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.v_head_dim, config.hidden_size, bias=config.attention_bias)
        self.rotary_emb = DeepseekV2YarnRotaryEmbedding(config)

    def forward(self, h: Tensor, position_ids: Tensor, attention_params: AttentionParameters) -> Tensor:
        n_tokens, hidden_size = h.shape
        q = self.q_proj(h) # (n_tokens, n_heads * (qk_nope_head_dim + qk_rope_head_dim)
        q = q.view(n_tokens, self.config.num_attention_heads, self.config.qk_nope_head_dim + self.config.qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.config.qk_nope_head_dim, self.config.qk_rope_head_dim], dim=-1) # (n_tokens, n_heads, config.qk_nope_head_dim) (num_attention_heads, n_tokens, qk_rope_head_dim)
        compressed_kv, k_pe = torch.split(self.kv_a_proj_with_mqa(h), [self.config.kv_lora_rank, self.config.qk_rope_head_dim], dim=-1) # (n_tokens, kv_lora_rank) (n_tokens, qk_rope_head_dim)
        q_pe, k_pe = self.rotary_emb(q_pe, k_pe)
        k_b_proj_weight, v_b_proj_weight = self.kv_b_proj.weight.split([self.config.num_attention_heads * self.config.qk_nope_head_dim, self.config.num_attention_heads, self.config.v_head_dim], dim=-1)
        k_b_proj_weight = k_b_proj_weight.view(self.config.num_attention_heads, self.config.qk_nope_head_dim, self.config.kv_lora_rank)
        v_b_proj_weight = v_b_proj_weight.view(self.config.num_attention_heads, self.config.v_head_dim, self.config.kv_lora_rank)
        q_nope_absorbed = torch.einsum("ijk,jko->ijo", q_nope, self,k_b_proj_weight) # (n_tokens, n_heads, kv_lora_rank)

        kv_cache: CompressedKVCache = attention_params.kv_cache
        kv_cache.set_kv_cache(slot_ids=attention_params.new_cache_slots, compressed_kv=compressed_kv, k_pe=k_pe)

        import flashinfer
        mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=q.device), 
            backend="fa2"
        )
        import math
        mla_wrapper.plan(
            qo_indptr = attention_params.q_cu_seq_lens, 
            kv_indptr = attention_params.block_tables, 
            kv_indices = attention_params.cu_blocks_lens, 
            kv_len_arr = attention_params.kv_seq_lens, 
            num_heads = self.config.num_attention_heads,
            head_dim_ckv = self.config.kv_lora_rank, 
            head_dim_kpe = self.config.qk_rope_head_dim,
            page_size = attention_params.block_size,
            causal = True,
            sm_scale = 1 / math.sqrt(self.config.qk_nope_head_dim + self.config.kv_lora_rank),
            q_data_type = q.dtype, 
            kv_data_type = q.dtype, 
            use_profiler = False, 
        )
        o = mla_wrapper.run(q_nope_absorbed, q_pe, kv_cache.cache[:, :, 0, :self.config.kv_lora_rank], k_pe[:, :, 0, self.config.kv_lora_rank:], return_lse=False) # (n_tokens, n_heads, kv_lora_rank)
        o = torch.einsum("ijk,jok->ijo", o, v_b_proj_weight).view(n_tokens, self.config.num_attention_heads * self.config.v_head_dim)
        o = self.o_proj(o) # (n_tokens, hidden_size)
        return o


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: DeepSeekV2Config, layer_id: int):
        super().__init__()
        self.self_attn = DeepseekV2Attention(config)
        if layer_id >= config.first_k_dense_replace and layer_id % config.moe_layer_freq:
            self.mlp = DeepseekV2MoE(config)
        else:
            self.mlp = DeepseekV2MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, h: Tensor) -> Tensor:
        # h (n_tokens, hidden_size)
        h = h + self.self_attn(self.input_layernorm(h))
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h

class DeepseekV2Model(nn.Module):
    def __init__(self, config: DeepSeekV2Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepseekV2DecoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: Tensor) -> Tensor:
        h = self.embed_tokens(input_ids) # (n_tokens, hidden_size)
        for layer in self.layers:
            h = layer(h) # (n_tokens, hidden_size)
        h = self.norm(h) # (n_tokens, hidden_size)
        return h

class DeepseekV2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DeepseekV2Model()
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids (n_tokens, )
        hidden_states = self.model(input_ids) # (n_tokens, hidden_size)
        logits = self.lm_head(hidden_states) # (n_tokens, vocab_size)
        return logits


if __name__ == '__main__':

    # json_path = "/mnt/cfs/9n-das-admin/llm_models/DeepSeek-V2-Lite/config.json"
    # # with open('/mnt/cfs/9n-das-admin/llm_models/DeepSeek-V2-Lite'):

    # import json
    # import dacite
    # with open(json_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)

    # config = dacite.from_dict(DeepSeekV2Config, data, config=dacite.Config(strict=False))
    # print(config)
    l1 = nn.Linear(10, 20)
    print(l1.weight.shape)