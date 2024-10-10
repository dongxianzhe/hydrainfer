import torch
from torch import nn, Tensor
from dxz.model.parameters import InputParameters

class KVCache:
    # todo
    pass

class Attention(nn.Module):
    def __init__(n_qo_head: int, n_kv_heads: int, head_dim: int, handler: str, sliding_window:int=-1):
        assert n_qo_head % n_kv_heads == 0, f"n_qo_heads {n_qo_head} is not divisible by n_kv_heads {n_kv_heads}"
        # todo handler
        # todo sliding window
        self.n_qo_head      = n_qo_head
        self.n_kv_heads     = n_kv_heads
        self.head_dim       = head_dim
        self.handler        = handler
        self.sliding_window = sliding_window

    def forward(query: Tensor, key: Tensor, value: Tensor, positions: Tensor, kv_cache: KVCache, input_params: InputParameters):
        # query (n_tokens, n_qo_heads * head_dim)
        # key/value (n_tokens, n_kv_heads * head_dim)
        # position (n_tokens, )
        # return (n_tokens, n_heads, head_dim)
        # 1. apply positional embedding
        # 2. append kv cache
        # 3. o = softmax(mask(alibi_bias(logits_soft_cap(sm_scale(q@k))))@v
        n_tokens = query.shape[0]
        q = query.view(n_tokens, self.n_qo_head, self.head_dim)
        k = query.view(n_tokens, self.n_kv_head, self.head_dim)
        v = query.view(n_tokens, self.n_kv_head, self.head_dim)

        q,k = handler.apply_pos_emb(q, k, positions)

        handler.append_kv_cache(kv_cache, k, v, input_params)

        o = torch.empty_like(q)
        if input_params.empty_kv_cache:
            handler.batch_prefill(q, k, v, input_params, self.sliding_window, o)
        else:
            handler.batch_decode(q, k, v, input_params, self.sliding_window, o)
        return o.view(n_tokens, -1)