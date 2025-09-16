"""
Deepseek V3 model with shared experts.

Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modular_deepseek_v3.py#L328
Apache-2.0 license
"""
from typing import Optional
import os
import safetensors
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
from hydrainfer.transformers_utils.deepseek_vl2_config import DeepseekVLV2Config, DeepseekV2Config
from hydrainfer.model.parameters import LanguageModelParameters, AttentionParameters
from hydrainfer.layer.norm import RMSNorm
from hydrainfer.layer.causal_attention import CausalGroupedQueryPageAttention, CausalGroupedQueryPageAttentionConfig
from hydrainfer.layer.rotary_embedding import RotaryEmbedding, compute_default_inv_freq
from hydrainfer.utils.logger import getLogger
from hydrainfer.model.model_forward import DecoderLayer, ROPECausalGroupedQueryPageAttention, GateUpDownMLP
logger = getLogger(__name__)


class DeepseekV3MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.mlp = GateUpDownMLP(gate_proj = self.gate_proj, up_proj = self.up_proj, down_proj = self.down_proj, activation = ACT2FN[config.hidden_act])

    def forward(self, h: Tensor) -> Tensor:
        return self.mlp.forward(h)

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )

    def forward(self, hidden_states):
        h = hidden_states.shape[-1]
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        else:
            raise NotImplementedError(
                f"insupportable topk method for MoE gating: {self.topk_method}"
            )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator * self.routed_scaling_factor
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight, 

class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = MoEGate(config)
        self.shared_experts = DeepseekV3MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class DeepseekV3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        config.head_dim = config.hidden_size // config.num_attention_heads
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
        self.attention = ROPECausalGroupedQueryPageAttention(q_proj = self.q_proj, k_proj = self.k_proj, v_proj = self.v_proj, o_proj = self.o_proj, rotary_emb = self.rotary_emb, n_qo_heads = config.num_attention_heads, n_kv_heads = config.num_key_value_heads, head_dim = config.head_dim)
        
    
    def forward(self, hidden_states: Tensor, position_ids: Tensor, attention_param: AttentionParameters) -> Tensor:
        return self.attention.forward(hidden_states, position_ids, attention_param)


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_idx
        self.self_attn = DeepseekV3Attention(config=config)
        self.mlp = DeepseekV3MoE(config) if layer_idx >= config.first_k_dense_replace else DeepseekV3MLP(config)
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

        self.decoder_layer = DecoderLayer(attention = self.self_attn, mlp = self.mlp, norm_1 = self.input_layernorm, norm_2 = self.post_attention_layernorm, layer_id = self.layer_id, n_layers = config.num_hidden_layers)


    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, model_params: LanguageModelParameters) -> Tensor:
        return self.decoder_layer.forward(hidden_states, position_ids, model_params)


class DeepseekV3Model(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: Tensor, position_ids: Tensor, model_params: LanguageModelParameters) -> Tensor:
        if input_ids.dtype == torch.int:
            # input_ids is token_ids
            hidden_states = self.embed_tokens(input_ids)
        else:
            # input_ids is input embeds
            hidden_states = input_ids

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_ids, model_params)
        return self.norm(hidden_states)

class DeepseekForCausalLM(nn.Module):
    def __init__(self, config: DeepseekV2Config, path: str):
        super().__init__()
        self.config = config
        self.config.image_token_id = AutoTokenizer.from_pretrained(path).vocab["<image>"]
        self.model = DeepseekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> Tensor:
        # input_ids (n_tokens, )
        # input_ids      (n_text_tokens + n_image_tokens) n_text_tokens is number of text tokens, n_image_tokens is number of image tokens
        # image_features (n_image_tokens, hidden_size)
        # position_ids (n_text_tokens + n_image_tokens)
        input_embeds = self.model.embed_tokens(input_ids) # (n_tokens, hidden_size)
        if image_features is not None:
            image_overwrite_mask = input_ids == self.config.image_token_id
            input_embeds[image_overwrite_mask, :] = image_features.view(-1, input_embeds.shape[-1])
        
        hidden_state = self.model(input_embeds, position_ids, model_params) # hidden_state (n_selected_tokens, hidden_size) we discard tokens that do not need to be sampled before entering into the last ffn layer to reduce redundant computation
        logits = self.lm_head(hidden_state) # (n_selected_tokens, hidden_size)
        sample_token_ids = torch.argmax(logits, dim=-1, keepdim=False) # (n_selected_tokens, )
        return sample_token_ids
    
    @classmethod
    def from_safetensor(cls, model_weights_path: str, dtype: torch.dtype, device: torch.device):
        # 1. create model
        config = DeepseekVLV2Config.from_pretrained(model_weights_path).text_config
        torch.set_default_dtype(dtype)
        with torch.device(device):
            model = cls(config, model_weights_path)
        torch.set_default_dtype(torch.float)

        # 2. load weights
        state_dict = model.state_dict()
        loaded_set = set()
        for entry in os.scandir(model_weights_path):
            if entry.is_file() and os.path.splitext(entry.name)[1] == '.safetensors':
                logger.info(f'load safetensor from {entry.path}')
                for name, weight in safetensors.torch.load_file(entry.path).items():
                    if name.startswith('language.'):
                        name = name.removeprefix('language.')
                        state_dict[name].copy_(weight)
                        loaded_set.add(name)
        model.load_state_dict(state_dict)
        model.to(dtype)
        model.eval()

        # 3. verify
        assert len(loaded_set) == len(state_dict), f"missing {state_dict.keys() - loaded_set} in safetensor"

        return model