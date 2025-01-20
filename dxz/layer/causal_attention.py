import math
import torch
from torch import nn, Tensor
from dxz.model.parameters import AttentionParameters
from dxz.utils.statistic import attention_score_heatmap, histogram
from dxz.utils import attention_utils
from dxz.utils.attention_utils import sparsity
from dataclasses import dataclass

try:
    from dxz._C.kernel.flash_attn import mha_varlen_fwd
except ImportError:
    print('self compiled flash attention mha_varlen_fwd import failed')
    mha_varlen_fwd = None


"""
we use chain of responsibility to design attention module considering the following requirements:
1. some kernel may be unavailable because of compilation failure, library installation failure.
2. some optimization need attention score to evict kvcache or prune token but fused kernel can't get intermediate attention score.
3. some kernel can't work on all dtype, hardware version.
4. we may want to dynamically select best kernel according to batch information and schedule policy
5. we may want to statically configure a specific attention kernel
so we design attention handlers for each kernel, and link them to a linked list data structure, each handler will check weather it can cope with the situation if it can, it will compute and return the result, if it can't, it will call the next handler to cope with it. at last the request must be handled becuase we have an pytorch implemented handler, however it is very slow.
"""


@dataclass
class CausalGroupedQueryPageAttentionConfig:
    n_qo_heads: int
    n_kv_heads: int
    head_dim: int


@dataclass
class CausalGroupedQueryPageAttentionOutput:
    o: Tensor


class FlashInferCausalGroupedQueryPageAttentionHandler(nn.Module):
    def __init__(self, config: CausalGroupedQueryPageAttentionConfig):
        super().__init__()
        assert config.n_qo_heads % config.n_kv_heads == 0, f"n_qo_heads {config.n_qo_heads} is not divisible by n_kv_heads {config.n_kv_heads}"
        self.n_qo_heads     = config.n_qo_heads
        self.n_kv_heads     = config.n_kv_heads
        self.head_dim       = config.head_dim

        self.next_handler: nn.Module = None

    def forward(self, query: Tensor, attention_params: AttentionParameters) -> CausalGroupedQueryPageAttentionOutput:
        # query (n_tokens, n_qo_heads * head_dim)
        # o (n_tokens, n_heads, head_dim)
        if query.device == torch.device('cpu'):
            return self.next_handler(query, attention_params)

        if attention_params.flash_infer_handler is None:
            return self.next_handler(query, attention_params)

        kv_cache = attention_params.kv_cache
        key_cache, value_cache = kv_cache.get_kv_cache()
        output = attention_params.flash_infer_handler.run(query, (key_cache, value_cache))
        output = output.view(-1, self.n_qo_heads * self.head_dim)

        return CausalGroupedQueryPageAttentionOutput(o = output)


class FlashAttentionCausalGroupedQueryPageAttentionHandler(nn.Module):
    def __init__(self, config: CausalGroupedQueryPageAttentionConfig):
        super().__init__()
        assert config.n_qo_heads % config.n_kv_heads == 0, f"n_qo_heads {config.n_qo_heads} is not divisible by n_kv_heads {config.n_kv_heads}"
        self.n_qo_heads     = config.n_qo_heads
        self.n_kv_heads     = config.n_kv_heads
        self.head_dim       = config.head_dim

        self.next_handler: nn.Module = None

    def forward(self, query: Tensor, attention_params: AttentionParameters) -> CausalGroupedQueryPageAttentionOutput:
        # query (n_tokens, n_qo_heads * head_dim)
        # o (n_tokens, n_heads, head_dim)
        if query.device == torch.device('cpu'):
            return self.next_handler(query, attention_params)
        if mha_varlen_fwd is None:
            return self.next_handler(query, attention_params)

        kv_cache = attention_params.kv_cache
        key_cache, value_cache = kv_cache.get_kv_cache()
        if attention_params.flash_infer_handler:
            output = attention_params.flash_infer_handler.run(query, (key_cache, value_cache))
            return output.view(-1, self.n_qo_heads * self.head_dim)

        output=torch.empty_like(query)
        mha_varlen_fwd(
            output,
            query, 
            key_cache, 
            value_cache, 
            attention_params.q_cu_seq_lens, 
            attention_params.kv_cu_seq_lens, 
            attention_params.block_tables, 
            attention_params.cu_blocks_lens, 
            None,
            attention_params.q_max_seq_len,
            attention_params.kv_max_seq_len,
            1. / math.sqrt(self.head_dim),
            0,
            -1,
            0,
            0
        )
        output = output.view(-1, self.n_qo_heads * self.head_dim)

        return CausalGroupedQueryPageAttentionOutput(o = output)


class TorchCausalGroupedQueryPageAttention(nn.Module):
    def __init__(self, config: CausalGroupedQueryPageAttentionConfig):
        super().__init__()
        assert config.n_qo_heads % config.n_kv_heads == 0, f"n_qo_heads {config.n_qo_heads} is not divisible by n_kv_heads {config.n_kv_heads}"
        self.n_qo_heads     = config.n_qo_heads
        self.n_kv_heads     = config.n_kv_heads
        self.head_dim       = config.head_dim

        self.next_handler: nn.Module = None

    def forward(self, query: Tensor, attention_params: AttentionParameters) -> CausalGroupedQueryPageAttentionOutput:
        # query (n_tokens, n_qo_heads * head_dim)
        # o (n_tokens, n_heads, head_dim)
        kv_cache = attention_params.kv_cache
        key_cache, value_cache = kv_cache.get_kv_cache()

        outputs = []
        for i in range(attention_params.num_sequences):
            block_table = attention_params.block_tables[attention_params.cu_blocks_lens[i]: attention_params.cu_blocks_lens[i + 1]]
            key = key_cache[block_table, :, :, :].reshape(-1, self.n_kv_heads, self.head_dim)
            value = value_cache[block_table, :, :, :].reshape(-1, self.n_kv_heads, self.head_dim)
            k = key[: attention_params.kv_cu_seq_lens[i + 1] - attention_params.kv_cu_seq_lens[i], :, :].to(torch.float)
            v = value[: attention_params.kv_cu_seq_lens[i + 1] - attention_params.kv_cu_seq_lens[i],:, :].to(torch.float)
            q = query[attention_params.q_cu_seq_lens[i]: attention_params.q_cu_seq_lens[i + 1], :, :].to(torch.float)
            # q (qo_seq_len, n_qo_heads, head_dim)
            # k (kv_seq_len, n_kv_heads, head_dim)
            # v (kv_seq_len, n_kv_heads, head_dim)
            group_size = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeats=group_size, dim=1)
            v = v.repeat_interleave(repeats=group_size, dim=1)
            # k (kv_seq_len, n_qo_heads, head_dim)
            # v (kv_seq_len, n_qo_heads, head_dim)

            # compute score
            scores = torch.einsum('qhd,khd->hqk', q, k)

            # sm_scale
            sm_scale = 1. / math.sqrt(self.head_dim)
            scores *= sm_scale

            # mask
            num_heads, q_seq_len, k_seq_len = scores.shape
            x = torch.arange(k_seq_len, device=q.device)[None, None, :].repeat(num_heads, q_seq_len, 1)
            y = torch.arange(q_seq_len, device=q.device)[None, :, None].repeat(num_heads, 1, k_seq_len)
            mask = (x - y) > (k_seq_len - q_seq_len)
            scores.masked_fill_(mask=mask, value=float('-inf'))

            # # save attention score data
            # if scores.shape[1] == scores.shape[2]:
            #     torch.save(scores, f'layer{attention_params.layer_id}.pt')

            # # plot atention score
            # if scores.shape[1] == scores.shape[2]: # prefill stage
            #     print(f'layer{attention_params.layer_id} scores.shape {scores.shape}')

            #     attention_utils.sparsity(scores, mask, 0)
            #     for head_id in range(scores.shape[0]):
            #         attention_score_heatmap(score=scores[head_id], name=f'attention_score_layer{attention_params.layer_id}_head{head_id}', fig_size=50)

            # # meausure sparsity
            # if scores.shape[1] == scores.shape[2]:
            #     print(f'sparsity {sparsity(scores, mask, 0.01)}')

            # # plot score distrubtion
            # if scores.shape[1] == scores.shape[2]:
            #     print(f'scores.shape {scores.shape}')
            #     histogram(data=scores, fig_size=5, range=(-25, 10), bins=100, name=f'distribution_attention_score_layer{attention_params.layer_id}')

            # softmax
            scores = torch.softmax(scores, dim=-1)
            o = torch.einsum("hqk,khd->qhd", scores, v)
            outputs.append(o)

        output = torch.cat(outputs, dim=0)
        output = output.view(-1, self.n_qo_heads * self.head_dim)
        output = output.to(query.dtype)

        return CausalGroupedQueryPageAttentionOutput(o = output)


class CausalGroupedQueryPageAttention(nn.Module):
    def __init__(self, config: CausalGroupedQueryPageAttentionConfig):
        super().__init__()
        assert config.n_qo_heads % config.n_kv_heads == 0, f"n_qo_heads {config.n_qo_heads} is not divisible by n_kv_heads {config.n_kv_heads}"
        self.n_qo_heads     = config.n_qo_heads
        self.n_kv_heads     = config.n_kv_heads
        self.head_dim       = config.head_dim

        self.handlers = [
            FlashInferCausalGroupedQueryPageAttentionHandler(config), 
            FlashAttentionCausalGroupedQueryPageAttentionHandler(config), 
            TorchCausalGroupedQueryPageAttention(config), 
        ]
        for i in range(len(self.handlers) - 1):
            self.handlers[i].next_handler = self.handlers[i + 1]
        self.handler = self.handlers[0]

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_params: AttentionParameters) -> CausalGroupedQueryPageAttentionOutput:
        # 1. reshape queyy key value shape
        n_tokens = query.shape[0]
        query = query.view(n_tokens, self.n_qo_heads, self.head_dim)
        key = key.view(n_tokens, self.n_kv_heads, self.head_dim)
        value = value.view(n_tokens, self.n_kv_heads, self.head_dim)

        # 2. append new kv cache
        kv_cache = attention_params.kv_cache
        kv_cache.set_kv_cache(attention_params.new_cache_slots, key,value)

        # 3. compute attention
        return self.handler(query, attention_params)