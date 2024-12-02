import torch
from torch import nn, Tensor
from dxz.model.parameters import AttentionParameters
from dxz.memory.kv_cache import KVCache
import math
from dxz.utils.statistic import attention_score_heatmap, histogram
from dxz.utils import attention_utils
from dxz.utils.attention_utils import sparsity

try :
    import flash_attn
except ImportError:
    print('flash attention import failed')
    flash_attn = None

try:
    from dxz._C.kernel.flash_attn import mha_varlen_fwd
except ImportError:
    print('flash attention mha_varlen_fwd import failed')
    mha_varlen_fwd = None

class TorchMultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, return_scores=False) -> Tensor:
        # query/key/value (batch_size, seq_len, hidden_size)
        # return (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = query.shape
        dtype = query.dtype

        query = query.view(-1, seq_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous().to(torch.float)
        key   =   key.view(-1, seq_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous().to(torch.float)
        value = value.view(-1, seq_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous().to(torch.float)
        query = query.view(-1, seq_len, self.head_dim)
        key   =   key.view(-1, seq_len, self.head_dim)
        value = value.view(-1, seq_len, self.head_dim)
        query *= 1. / math.sqrt(self.head_dim)
        score = torch.bmm(query, key.transpose(1, 2)) # (batch_size * n_heads, seq_len, seq_len)
        ret_score = score
        score = torch.softmax(score, dim=-1) # (batch_size * n_heads, seq_len, seq_len)
        o = torch.bmm(score, value) # (batch_size * n_heads, seq_len, head_dim)
        o = o.view(batch_size, self.n_heads, seq_len, self.head_dim).transpose(1, 2).contiguous() # (batch_size, seq_len, n_heads, head_dim)
        o = o.view(batch_size, seq_len, hidden_size).to(dtype) # (batch_size, seq_len, hidden_size)

        if return_scores:
            return o, ret_score
        else:
            return o

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # query/key/value (batch_size, seq_len, hidden_size)
        # return (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = query.shape
        device = query.device
        dtype = query.dtype

        # 1. try to use flash attention
        if flash_attn:
            query = query.view(batch_size, seq_len, self.n_heads, self.head_dim) # (batch_size * seq_len, n_heads, head_dim)
            key   =   key.view(batch_size, seq_len, self.n_heads, self.head_dim) # (batch_size * seq_len, n_heads, head_dim)
            value = value.view(batch_size, seq_len, self.n_heads, self.head_dim) # (batch_size * seq_len, n_heads, head_dim)
            o = flash_attn.flash_attn_func(
                q = query,
                k = key,
                v = value,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
                window_size=(-1, -1),  # -1 means infinite context window
                softcap=0.0, # 0.0 means deactivated
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=False,
            )
            o = o.view(batch_size, seq_len, hidden_size)
            return o
        # 2. try to use my flash attention
        query = query.view(batch_size * seq_len, self.n_heads, self.head_dim) # (batch_size * seq_len, n_heads, head_dim)
        key   =   key.view(batch_size * seq_len, self.n_heads, self.head_dim) # (batch_size * seq_len, n_heads, head_dim)
        value = value.view(batch_size * seq_len, self.n_heads, self.head_dim) # (batch_size * seq_len, n_heads, head_dim)
        o = torch.empty(size=(batch_size * seq_len, self.n_heads, self.head_dim), dtype=dtype, device=device)
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int, device=device)
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int, device=device)
        mha_varlen_fwd(
            o, 
            query, 
            key, 
            value, 
            cu_seqlens_q, 
            cu_seqlens_k,
            None, 
            None, 
            None, 
            seq_len, 
            seq_len, 
            1. / math.sqrt(self.head_dim),
            0, 
            -1,
            -1, 
            0, 
        )

        return o.view(batch_size, seq_len, hidden_size)

class FlashCausalGroupedQueryPageAttention(nn.Module):
    def __init__(self, n_qo_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        assert n_qo_heads % n_kv_heads == 0, f"n_qo_heads {n_qo_heads} is not divisible by n_kv_heads {n_kv_heads}"
        self.n_qo_heads     = n_qo_heads
        self.n_kv_heads     = n_kv_heads
        self.head_dim       = head_dim

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_params: AttentionParameters):
        # query (n_tokens, n_qo_heads * head_dim)
        # key/value (n_tokens, n_kv_heads * head_dim)
        # return (n_tokens, n_heads, head_dim)

        # 1. reshape queyy key value shape
        n_tokens = query.shape[0]
        query = query.view(n_tokens, self.n_qo_heads, self.head_dim)
        key = key.view(n_tokens, self.n_kv_heads, self.head_dim)
        value = value.view(n_tokens, self.n_kv_heads, self.head_dim)

        # 2. append new kv cache
        kv_cache = attention_params.kv_cache
        key_cache, value_cache = kv_cache.get_kv_cache()
        kv_cache.set_kv_cache(
            attention_params.new_cache_slots, # slot_ids: Tensor,  # [n_tokens]
            key, # keys: Tensor,      # [n_tokens, n_kv_heads, head_dim]
            value, # values: Tensor,    # [n_tokens, n_kv_heads, head_dim]
        )

        # 3. compute for each sequence with flash attn
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
        return output.view(-1, self.n_qo_heads * self.head_dim)

class TorchCausalGroupedQueryPageAttention(nn.Module):
    def __init__(self, n_qo_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        assert n_qo_heads % n_kv_heads == 0, f"n_qo_heads {n_qo_heads} is not divisible by n_kv_heads {n_kv_heads}"
        self.n_qo_heads     = n_qo_heads
        self.n_kv_heads     = n_kv_heads
        self.head_dim       = head_dim

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_params: AttentionParameters):
        # query (n_tokens, n_qo_heads * head_dim)
        # key/value (n_tokens, n_kv_heads * head_dim)
        # return (n_tokens, n_heads, head_dim)
        # 1. reshape queyy key value shape
        n_tokens = query.shape[0]
        query = query.view(n_tokens, self.n_qo_heads, self.head_dim)
        key = key.view(n_tokens, self.n_kv_heads, self.head_dim)
        value = value.view(n_tokens, self.n_kv_heads, self.head_dim)

        # 2. append new kv cache
        kv_cache = attention_params.kv_cache
        key_cache, value_cache = kv_cache.get_kv_cache()
        kv_cache.set_kv_cache(attention_params.new_cache_slots, key,value)

        # 3. compute for each sequence with pytorch
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
        return output.to(query.dtype)