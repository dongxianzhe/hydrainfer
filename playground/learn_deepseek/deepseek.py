import os
import json
import safetensors
import safetensors.torch
import torch
from torch import nn, Tensor
from itertools import accumulate 
from dataclasses import dataclass
from hydrainfer.model_parallel.process_group import ProcessGroup
from hydrainfer.layer.norm import RMSNorm
from hydrainfer.layer.linear import ColumnParallelLinear, RowParallelLinear
from hydrainfer.layer.causal_attention import CausalGroupedQueryPageAttention, CausalGroupedQueryPageAttentionConfig
from hydrainfer.layer.rotary_embedding import RotaryEmbedding, compute_default_inv_freq
from hydrainfer.model.parameters import LanguageModelParameters, LanguageModelOutput, AttentionParameters
from hydrainfer.layer.activation import silu



@dataclass
class DeepseekV2Config:
    vocab_size: int
    hidden_size: int
    rms_norm_eps: float
    num_hidden_layers: int
    enable_mla: bool
    n_routed_experts: int
    first_k_dense_replace: int
    moe_layer_freq: int
    q_lora_rank: int
    n_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    kv_lora_rank: int
    v_head_dim: int
    head_dim: int # deprecated
    intermediate_size: int
    topk_method: str
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    n_shared_experts: int
    routed_scaling_factor: int

@dataclass
class ParallelConfig:
    dp_size: int
    ep_size: int
    global_process_group: ProcessGroup
    dp_local_process_group: ProcessGroup


class DeepseekV2Attention(nn.Module):
    def __init__(self, model_config: DeepseekV2Config, parallel_config: ParallelConfig):
        super().__init__()
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.qk_head_dim = model_config.qk_nope_head_dim + model_config.qk_rope_head_dim
        if model_config.q_lora_rank > 0:
            self.q_a_proj =  nn.Linear(model_config.hidden_size, model_config.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(model_config.q_lora_rank, model_config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                in_features = model_config.q_lora_rank, 
                out_features = model_config.n_heads * self.qk_head_dim, 
                bias=False, 
                gather_output=False, 
                process_group=parallel_config.dp_local_process_group,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                in_features = model_config.hidden_size, 
                out_features = model_config.n_heads * self.qk_head_dim, 
                bias=False, 
                gather_output=False, 
                process_group=parallel_config.dp_local_process_group, 
            )
        self.kv_a_proj_with_mqa = nn.Linear(model_config.hidden_size, model_config.kv_lora_rank + model_config.qk_rope_head_dim)
        self.kv_a_layernorm = RMSNorm(model_config.kv_lora_rank, model_config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            in_features = model_config.kv_lora_rank, 
            out_features = model_config.n_heads * (model_config.qk_nope_head_dim + model_config.v_head_dim), 
            bias = False, 
            gather_output = False, 
            process_group = self.parallel_config.dp_local_process_group, 
        )
        self.o_proj = RowParallelLinear(
            in_features = model_config.n_heads * model_config.v_head_dim, 
            out_features = model_config.hidden_size, 
            bias = False, 
            input_is_parallelized = True, 
            reduce_output= True, 
            process_group = self.parallel_config.dp_local_process_group, 
        )
        self.n_local_heads = model_config.n_heads // self.parallel_config.dp_local_process_group.world_size, 
        self.attention = CausalGroupedQueryPageAttention(
            config=CausalGroupedQueryPageAttentionConfig(
                n_qo_heads = self.n_local_heads, 
                n_kv_heads = self.n_local_heads, 
                head_dim = model_config.head_dim, 
            )
        )

        self.rotary_embed = RotaryEmbedding(
            rotary_dim = model_config.head_dim, 
            max_position_embeddings = model_config.max_position_embeddings, 
            inv_freq=compute_default_inv_freq(
                rotary_dim=model_config.head_dim, 
                theta=model_config.rope_theta, 
            ), 
            interleaved = True
        )

    def forward(self, h: Tensor, position_ids: Tensor, attention_params: AttentionParameters) -> Tensor:
        # h (n_tokens_in_dp_group, hidden_size)
        if self.model_config.q_lora_rank > 0:
            q = self.q_a_proj(h) # (n_tokens_in_dp_group, hidden_size)
            q = self.q_a_layernorm(q) # (n_tokens_in_dp_group, hidden_size)
            q = self.q_b_proj(q) # (n_tokens_in_dp_group, n_local_heads * (qk_nope_head_dim + qk_rope_head_dim))
        else:
            q = self.q_proj(h) # (n_tokens_in_dp_group, n_local_heads * (qk_nope_head_dim + qk_rope_head_dim))
        q = q.view(-1, self.n_local_heads, self.qk_head_dim) # (n_tokens_in_dp_group, n_local_heads, qk_nope_head_dim + qk_rope_head_dim)
        q_pe = torch.split(q, [self.model_config.qk_nope_head_dim, self.model_config.qk_rope_head_dim], dim=-1)[1]

        latent_cache = self.kv_a_proj_with_mqa(h) # (n_tokens_in_dp_group, (kv_lora_rank + qk_rope_head_dim))
        latent_cache_split: list[Tensor] = torch.split(latent_cache, [self.model_config.kv_lora_rank, self.model_config.qk_rope_head_dim])
        kv_a = latent_cache_split[0] # (n_tokens_in_dp_group, kv_lora_rank)
        k_pe = latent_cache_split[1][:, None, :] # (n_tokens_in_dp_group, 1, qk_rope_head_dim)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a) # (n_tokens_in_dp_group, n_local_heads * (qk_nope_head_dim + v_head_dim))
        kv = kv.view(-1, self.n_local_heads, self.model_config.qk_nope_head_dim, self.model_config.v_head_dim) # (n_tokens_in_dp_group, n_local_heads, qk_nope_head_dim + v_head_dim)
        kv_split: list[Tensor] = torch.split(kv, [self.model_config.qk_nope_head_dim, self.model_config.v_head_dim], dim=-1)
        k_nope = kv_split[0] # (n_tokens_in_dp_group, n_local_heads, qk_nope_head_dim)
        v = kv_split[1] # (n_tokens_in_dp_group, n_local_heads, v_head_dim)

        q_pe, k_pe = self.rotary_embed(q_pe, k_pe, position_ids)
        q[:, :, self.model_config.qk_nope_head_dim:] = q_pe
        k = torch.empty_like(q)
        k[:, :, :self.model_config.qk_nope_head_dim] = k_nope
        k[:, :, self.model_config.qk_nope_head_dim:] = k_pe
        q = nn.functional.pad(q, (0, self.model_config.head_dim - self.qk_head_dim))
        k = nn.functional.pad(k, (0, self.model_config.head_dim - self.qk_head_dim))
        v = nn.functional.pad(v, (0, self.model_config.head_dim - self.model_config.v_head_dim))
        o = self.attention(q, k, v, attention_params).o
        o = o.view(-1, self.n_local_heads * self.model_config.v_head_dim) # (n_tokens_in_dp_group, n_local_heads * v_head_dim)
        o = self.o_proj(o) # (n_tokens_in_dp_group, hidden_size)
        return o


class DeepseekV2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, process_group: ProcessGroup, reduce_output: bool):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            in_features = hidden_size, 
            out_features = intermediate_size, 
            bias = False, 
            gather_output = False, 
            process_group = process_group, 
        )
        self.up_proj = ColumnParallelLinear(
            in_features = hidden_size, 
            out_features = intermediate_size, 
            bias = False, 
            gather_output = False, 
            process_group = process_group, 
        )
        self.down_proj = RowParallelLinear(
            in_features = intermediate_size, 
            out_features = hidden_size, 
            bias = False, 
            input_is_parallelized = True, 
            reduce_output = reduce_output, 
            process_group = process_group, 
        )

    def forward(self, x: Tensor):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(silu(gate) * up)

class MoEGate(nn.Module):
    def __init__(self, n_routed_experts: int, hidden_size: int, is_noaux_tc: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size=(n_routed_experts, hidden_size)), requires_grad=False)
        if is_noaux_tc:
            self.e_score_correction_bias = nn.Parameter(torch.empty(size=(n_routed_experts)), requires_grad=False)

    def forward(self, h: Tensor) -> Tensor:
        return nn.functional.linear(h, self.weight)
    
    def get_correction_bias(self) -> Tensor:
        return self.e_score_correction_bias


class EPMoE:
    def __init__(
        self, 
        num_experts: int, 
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool,
        use_grouped_topk: bool,
        num_expert_group: int,
        topk_group: int,
        correction_bias: Tensor,
        process_group: ProcessGroup, 
    ):
        assert num_experts % process_group.world_size == 0   
        num_experts_per_partition = num_experts // process_group.world_size
        ep_rank = process_group.rank
        start_expert_id = ep_rank * num_experts_per_partition
        end_expert_id = start_expert_id + num_experts_per_partition - 1


class DeepseekV2MoE(nn.Module):
    def __init__(self, model_config: DeepseekV2Config, process_group: ProcessGroup):
        super().__init__()
        self.model_config = model_config
        self.process_group = process_group
        assert model_config.n_routed_experts > process_group.world_size
        is_noaux_tc: bool = model_config.topk_method == 'noaux_tc'
        self.gate = MoEGate(
            model_config.n_routed_experts, 
            model_config.hidden_size, 
            is_noaux_tc
        )
        self.experts = EPMoE(
            num_experts = model_config.n_routed_experts, 
            top_k = model_config.num_experts_per_tok,
            hidden_size = model_config.hidden_size,
            intermediate_size = model_config.moe_intermediate_size,
            renormalize = model_config.norm_topk_prob,
            use_grouped_topk = True,
            num_expert_group = model_config.n_group,
            topk_group = model_config.topk_group,
            correction_bias = self.gate.get_correction_bias(),
            process_group = process_group, 
        )
        if model_config.n_shared_experts > 0:
            self.shared_exprets = DeepseekV2MLP(
                hidden_size=model_config.hidden_size, 
                intermediate_size=model_config.hidden_size * model_config.n_shared_experts, 
                process_group = process_group, 
                reduce_output = False
            )

    def forward(self, h: Tensor) -> Tensor:
        shared_output = self.shared_exprets(h) # (n_tokens, hidden_size)
        router_logits = self.gate(h) # (n_tokens, n_experts)
        h = self.experts(h, router_logits) # (n_tokens, hidden_size)
        h = h * self.model_config.routed_scaling_factor
        if self.model_config.n_shared_experts > 0:
            h = h + shared_output
        if self.process_group > 0:
            h = self.process_group.reduce_from_tensor_parallel_region(h)
        return h


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, model_config: DeepseekV2Config, parallel_config: ParallelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.model_config = model_config
        self.parallel_config = parallel_config
        if model_config.enable_mla:
            self.self_attn = DeepseekV2AttentionMLA(model_config, parallel_config.dp_local_process_group, )
        else:
            self.self_attn = DeepseekV2Attention(model_config, parallel_config.dp_local_process_group)
        if model_config.n_routed_experts > 0 and layer_id > model_config.first_k_dense_replace and layer_id % model_config.moe_layer_freq == 0:
            self.mlp = DeepseekV2MoE(model_config, parallel_config)
        else:
            self.mlp = DeepseekV2MLP(model_config.hidden_size, model_config.intermediate_size, parallel_config.ep_process_group)
        self.input_layernorm = RMSNorm(model_config.hidden_size, model_config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(model_config.hidden_size, model_config.rms_norm_eps)

    def forward(self, h: Tensor, position_ids: Tensor, model_params: LanguageModelParameters, n_tokens_each_dp_group: list[int]) -> LanguageModelOutput:
        if h.numel() > 0:
            r = h
            h = self.input_layernorm(h)
            h = self.self_attn(h, position_ids, model_params.attention_params[self.layer_id])
            h = r + h

            r = h
            h = self.post_attention_layernorm(h)

        def all_gather_across_all_dp_groups(h: Tensor) -> Tensor:
            max_n_tokens = max(n_tokens_each_dp_group)
            # (n_tokens, hidden_size)
            if h is not None:
                padded_h = nn.functional.pad(h, (0, 0, 0, max_n_tokens - h.shape[0])) # (max_n_tokens, hidden_size)
            else:
                padded_h = torch.empty(size=(max_n_tokens, self.model_config.hidden_size), dtype=h.dtype, device=h.device)

            dp_local_tp_size: int = self.parallel_config.global_process_group.world_size // self.parallel_config.dp_size

            raw_gathered_h = self.parallel_config.global_process_group.gather_from_data_parallel_region(padded_h)

            squeezed_gathered_h: list[Tensor] = []
            for dp_rank in range(self.parallel_config.dp_size):
                squeezed_gathered_h.append(
                    raw_gathered_h.narrow(dim=0,start=dp_rank * dp_local_tp_size * max_n_tokens, length=n_tokens_each_dp_group[dp_rank]))
            return torch.cat(squeezed_gathered_h, dim=0)

        if self.parallel_config.dp_size > 1:
            h = all_gather_across_all_dp_groups(h)

        h = self.mlp(h)

        # split MoE results for next layer's attention dp computation
        def pick_single_dp_rank(h: Tensor):
            dp_local_tp_size = self.parallel_config.global_process_group.world_size // self.parallel_config.dp_size

            dp_group_id: int = self.parallel_config.global_process_group.rank // dp_local_tp_size
            sum_n_tokens_each_dp_group = accumulate(n_tokens_each_dp_group, initial=0)
            h = h[sum_n_tokens_each_dp_group[dp_group_id]:sum_n_tokens_each_dp_group[dp_group_id + 1]]
            return h

        if self.parallel_config.dp_size > 1:
            h = pick_single_dp_rank(h)

        h = r + h
        return h

class DeepseekV2Model(nn.Module):
    def __init__(self, model_config: DeepseekV2Config, parallel_config: ParallelConfig):
        super().__init__()
        self.embed_tokens = ParallelEmbedding(vocab_size=model_config.vocab_size, embed_dim=model_config.hidden_size, process_group=parallel_config.dp_local_process_group)
        self.layers = nn.ModuleList(
            [DeepseekV2DecoderLayer(layer_id, model_config, parallel_config) for layer_id in range(model_config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size=model_config.hidden_size, eps=model_config.rms_norm_eps)

    def forward(self, token_ids: Tensor, position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        assert token_ids.numel() > 0
        assert token_ids.dtype == torch.int
        h = self.embed_tokens(token_ids)
        for layer in self.layers:
            h = layer(h, position_ids, model_params)
        return self.norm(h)

class DeepseekV2ForCausalLM(nn.Module):
    def __init__(self, model_config: DeepseekV2Config, parallel_config: ParallelConfig):
        super().__init__()
        self.model = DeepseekV2Model(model_config, parallel_config)
        self.lm_head = ColumnParallelLinear(in_features=model_config.hidden_sizea, out_features=model_config.vocab_size, bias=False, gather_output=True, process_group=parallel_config.dp_local_process_group)

    def forward(self, token_ids: Tensor, position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        h = self.model(token_ids, position_ids, model_params) # (n_selected_tokens, hidden_size)
        logits = self.lm_head(h)
        sample_token_ids = torch.argmax(logits, dim=-1, keepdim=False) # (n_selected_tokens, )
        return LanguageModelOutput(sample_token_ids=sample_token_ids)


if __name__ == '__main__':
    model_path = '/mnt/cfs/9n-das-admin/llm_models/DeepSeek-R1'
    with open(os.path.join(model_path, 'config.json')) as file:
        config_data = json.load(file)
    print(json.dumps(config_data, indent=4, ensure_ascii=False))

    for entry in os.scandir(model_path):
        if entry.is_file and os.path.splitext(entry.name)[1] == '.safetensors':
            print(f'load safetensor from {entry.path}')
            for name, weight in safetensors.torch.load_file(entry.path).items():
                print(f'{name} {weight.shape}')