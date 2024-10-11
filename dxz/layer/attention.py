import torch
from torch import nn, Tensor
from dxz.model.parameters import InputParameters
from dxz.memory.kv_cache import KVCache
from dxz.layer.rotary_embedding import RotaryEmbedding
import math

class Attention(nn.Module):
    def __init__(self, n_qo_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        assert n_qo_heads % n_kv_heads == 0, f"n_qo_heads {n_qo_heads} is not divisible by n_kv_heads {n_kv_heads}"
        # todo handler
        # todo sliding window
        self.n_qo_heads     = n_qo_heads
        self.n_kv_heads     = n_kv_heads
        self.head_dim       = head_dim

    def forward(self, query: Tensor, key: Tensor, value: Tensor, kv_cache: KVCache, input_params: InputParameters):
        # query (n_tokens, n_qo_heads * head_dim)
        # key/value (n_tokens, n_kv_heads * head_dim)
        # return (n_tokens, n_heads, head_dim)
        # 1. reshape queyy key value shape
        # 2. rotary position embedding
        # 3. append kv cache page attention
        # 4. o = softmax(mask(alibi_bias(logits_soft_cap(sm_scale(q@k))))@v flash atten kernel todo

        # 1. reshape queyy key value shape
        n_tokens = query.shape[0]
        query = query.view(n_tokens, self.n_qo_heads, self.head_dim)
        key = key.view(n_tokens, self.n_kv_heads, self.head_dim)
        value = value.view(n_tokens, self.n_kv_heads, self.head_dim)

        # 2. append new kv cache
        input_params.to(torch.device('cpu'))
        key_cache, value_cache = kv_cache.get_kv_cache()
        block_size = kv_cache.block_size
        for i, slot_id in enumerate(input_params.new_cache_slots):
            block_id = slot_id // block_size
            block_offset = slot_id  % block_size
            key_cache[block_id, block_offset, :, :] = key[i, :, :]
            value_cache[block_id, block_offset, :, :] = value[i, :, :]
        
        outputs = []
        # 3. compute for each sequence
        for i in range(input_params.num_sequences):
            block_table = input_params.block_tables[:input_params.cu_blocks_lens[i + 1] - input_params.cu_blocks_lens[i]]
            key = key_cache[block_table, :, :, :].reshape(-1, self.n_kv_heads, self.head_dim)
            value = value_cache[block_table, :, :, :].reshape(-1, self.n_kv_heads, self.head_dim)
            k = key[: input_params.kv_cu_seq_lens[i + 1] - input_params.kv_cu_seq_lens[i], :, :]
            v = value[: input_params.kv_cu_seq_lens[i + 1] - input_params.kv_cu_seq_lens[i],:, :]
            q = query[input_params.q_cu_seq_lens[i]: input_params.q_cu_seq_lens[i + 1], :, :]
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
            mask = x - y > (k_seq_len - q_seq_len)
            scores = scores.masked_fill_(mask=mask, value=float('-inf'))

            # softmax
            scores = torch.softmax(scores, dim=-1)
            o = torch.einsum("hqk,khd->qhd", scores, v)
            outputs.append(o)

        output = torch.cat(outputs, dim=0)
        output = output.view(-1, self.n_qo_heads * self.head_dim)
        return output