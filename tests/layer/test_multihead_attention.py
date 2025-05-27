import torch
import pytest
import math

from torch import nn

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_lens", [[5], [4, 6], [3, 2, 7]])
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("n_heads", [2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda:0" if torch.cuda.is_available() else "cpu"])
@torch.inference_mode()
def test_Qwenattention(batch_size, seq_lens, head_dim, n_heads,dtype ,device):
    # config Handler
    from hydrainfer.layer.multihead_attention import MultiHeadAttentionConfig
    from hydrainfer.layer.multihead_attention import QwenTorchMultiHeadAttentionHandler
    from hydrainfer.layer.multihead_attention import QwenFlashAttentionMultiHeadAttentionHandler
    from hydrainfer.layer.multihead_attention import QwenFlashAttentionMutliHeadAttentionHandler2

    # Handler
    config = MultiHeadAttentionConfig(n_heads=n_heads, head_dim=head_dim)

    handler_flash2 = QwenFlashAttentionMutliHeadAttentionHandler2(config)
    handler_flash = QwenFlashAttentionMultiHeadAttentionHandler(config)
    handler_torch = QwenTorchMultiHeadAttentionHandler(config)

    handler_flash.next_handler = handler_torch
    # packed q/k/v (seq_length, n_heads, head_dim)
    total_seqlen = sum(seq_lens)
    hidden_size = n_heads * head_dim
    hidden_states = torch.randn(total_seqlen, hidden_size, dtype=dtype, device=device)
    qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True).to(device, dtype)
    qkv = qkv_proj(hidden_states)
    qkv = qkv.reshape(total_seqlen, 3, n_heads, head_dim).permute(1, 0, 2, 3)
    q, k, v = qkv.unbind(0)

    # q = torch.randn(total_seqlen, n_heads, head_dim, dtype=dtype, device=device)
    # k = torch.randn(total_seqlen, n_heads, head_dim, dtype=dtype, device=device)
    # v = torch.randn(total_seqlen, n_heads, head_dim, dtype=dtype, device=device)
    
    # cu_seqlens (cumulative sum, batch start index)
    cu_seqlens = [0]
    for l in seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + l)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # forward
    o_flash2 = handler_flash2(q, k, v, total_seqlen, cu_seqlens)
    o_flash = handler_flash(q, k, v, total_seqlen, cu_seqlens)
    o_torch = handler_torch(q, k, v, total_seqlen, cu_seqlens)
    
    print(f'o_flash[:5]: {o_flash.view(-1)[:5]}')
    print(f'o_torch[:5]: {o_torch.view(-1)[:5]}')
    
    assert o_flash.shape == o_torch.shape
    assert o_flash2.shape == o_torch.shape
    print("result judge")
    assert torch.allclose(o_flash, o_torch, atol=1e-2, rtol=1e-2), f"max diff={torch.max(torch.abs(o_flash - o_torch))}"
    assert torch.allclose(o_flash2, o_torch, atol=1e-2, rtol=1e-2), f"max diff={torch.max(torch.abs(o_flash2 - o_torch))}"


if __name__ == '__main__':
    pytest.main([__file__])