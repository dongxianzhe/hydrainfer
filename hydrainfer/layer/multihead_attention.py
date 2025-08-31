import math
import torch
from torch import Tensor, nn
from typing import Optional
from dataclasses import dataclass
from hydrainfer.utils.logger import getLogger
logger = getLogger(__name__)

try:
    import flash_attn
except ImportError:
    logger.warning('flash attention import failed')
    flash_attn = None

try:
    from hydrainfer._C.kernel.flash_attn import mha_varlen_fwd
except ImportError:
    logger.warning('self compiled flash attention mha_varlen_fwd import failed')
    mha_varlen_fwd = None


@dataclass
class MultiHeadAttentionConfig:
    n_heads: int
    head_dim: int
    causal: bool = False


@dataclass
class MultiHeadAttentionParameters:
    return_scores: bool = False


@dataclass
class MultiHeadAttentionOutput:
    o: Tensor
    attention_scores: Optional[Tensor]


class TorchMultiHeadAttentionHandler(nn.Module):
    def __init__(self, config: MultiHeadAttentionConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.next_handler: nn.Module = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, params: MultiHeadAttentionParameters) -> MultiHeadAttentionOutput:
        # query/key/value (batch_size, seq_len, hidden_size)
        # o (batch_size, seq_len, hidden_size)
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
        attention_scores = score.view(batch_size, self.n_heads, seq_len, seq_len)
        score = torch.softmax(score, dim=-1) # (batch_size * n_heads, seq_len, seq_len)
        o = torch.bmm(score, value) # (batch_size * n_heads, seq_len, head_dim)
        o = o.view(batch_size, self.n_heads, seq_len, self.head_dim).transpose(1, 2).contiguous() # (batch_size, seq_len, n_heads, head_dim)
        o = o.view(batch_size, seq_len, hidden_size).to(dtype) # (batch_size, seq_len, hidden_size)

        return MultiHeadAttentionOutput(
            o = o,
            attention_scores=attention_scores
        )

class FlashAttentionMultiHeadAttentionHandler(nn.Module):
    def __init__(self, config: MultiHeadAttentionConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.next_handler: nn.Module = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, params: MultiHeadAttentionParameters) -> MultiHeadAttentionOutput:
        # query/key/value (batch_size, seq_len, hidden_size)
        # o (batch_size, seq_len, hidden_size)
        if flash_attn is None:
            return self.next_handler(query, key, value, params)
        if params.return_scores:
            return self.next_handler(query, key, value, params)

        batch_size, seq_len, hidden_size = query.shape
        device = query.device
        dtype = query.dtype
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
        return MultiHeadAttentionOutput(
            o = o, 
            attention_scores = None, 
        )


class FlashAttentionMutliHeadAttentionHandler2(nn.Module):
    def __init__(self, config: MultiHeadAttentionConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.next_handler: nn.Module = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, params: MultiHeadAttentionParameters) -> MultiHeadAttentionOutput:
        if mha_varlen_fwd is None:
            return self.next_handler(query, key, value, params)
        if params.return_scores:
            return self.next_handler(query, key, value, params)

        batch_size, seq_len, hidden_size = query.shape
        device = query.device
        dtype = query.dtype
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

        o = o.view(batch_size, seq_len, hidden_size)
        return MultiHeadAttentionOutput(
            o = o, 
            attention_scores = None, 
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, config: MultiHeadAttentionConfig):
        super().__init__()
        self.handlers = [
            FlashAttentionMutliHeadAttentionHandler2(config), 
            FlashAttentionMultiHeadAttentionHandler(config), 
            TorchMultiHeadAttentionHandler(config), 
        ] 
        for i in range(len(self.handlers) - 1):
            self.handlers[i].next_handler = self.handlers[i + 1]
        self.handler = self.handlers[0]

    def forward(self, query: Tensor, key: Tensor, value: Tensor, params: MultiHeadAttentionParameters) -> MultiHeadAttentionOutput:
        return self.handler(query, key, value, params)


class QwenFlashAttentionMutliHeadAttentionHandler2(nn.Module):
    def __init__(self, config: MultiHeadAttentionConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.next_handler: nn.Module = None
    def forward(self, q: Tensor, k: Tensor, v: Tensor,seq_length ,cu_seqlens: torch.Tensor) -> torch.Tensor:
        if mha_varlen_fwd is None:
            return self.next_handler(q, k, v, seq_length,cu_seqlens)
        
        dtype = q.dtype
        device = q.device
        attn_output = torch.empty(size=(seq_length, self.n_heads, self.head_dim), dtype=dtype, device=device)
        mha_varlen_fwd(
            attn_output, 
            q, 
            k, 
            v, 
            cu_seqlens, 
            cu_seqlens,
            None, 
            None, 
            None, 
            seq_length, 
            seq_length, 
            1. / math.sqrt(self.head_dim),
            0, 
            -1,
            -1, 
            0, 
        )
        attn_output=attn_output.reshape(seq_length, -1)
        return attn_output

class QwenFlashAttentionMultiHeadAttentionHandler(nn.Module):
    def __init__(self, config: MultiHeadAttentionConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.next_handler: nn.Module = None
    def forward(self, q: Tensor, k: Tensor, v: Tensor,seq_length ,cu_seqlens: torch.Tensor) -> torch.Tensor:
        if flash_attn is None:
            return self.next_handler(q, k, v, seq_length,cu_seqlens)


        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        return attn_output

class QwenTorchMultiHeadAttentionHandler(nn.Module):
    def __init__(self, config: MultiHeadAttentionConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.next_handler: nn.Module = None
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor,seq_length ,cu_seqlens: torch.Tensor) -> torch.Tensor:
        # query/key/value (batch_size, seq_len, hidden_size)
        attention_mask = torch.full([1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0            
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)

        return attn_output

class QwenMultiHeadAttention(nn.Module):
    def __init__(self, config: MultiHeadAttentionConfig):
        super().__init__()
        self.handlers = [
            QwenFlashAttentionMutliHeadAttentionHandler2(config), 
            QwenFlashAttentionMultiHeadAttentionHandler(config), 
            QwenTorchMultiHeadAttentionHandler(config), 
        ] 
        for i in range(len(self.handlers) - 1):
            self.handlers[i].next_handler = self.handlers[i + 1]
        self.handler = self.handlers[0]

    def forward(self, query: Tensor, key: Tensor, value: Tensor,seq_length ,cu_seqlens: torch.Tensor) -> MultiHeadAttentionOutput:
        return self.handler(query, key, value,seq_length,cu_seqlens)
