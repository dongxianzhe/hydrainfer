import torch
import pytest
from hydrainfer.layer.multihead_attention import TorchMultiHeadAttentionHandler, FlashAttentionMultiHeadAttentionHandler, FlashAttentionMutliHeadAttentionHandler2, MultiHeadAttentionConfig, MultiHeadAttentionParameters


@pytest.mark.parametrize("batch_size", [1, 4, 15, 16])
@pytest.mark.parametrize("seq_len", [1, 14, 111, 576])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("n_heads", [8, 4, 2, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
@torch.inference_mode()
def test_attention(
    batch_size: int, 
    seq_len: int, 
    head_dim: int, 
    n_heads: int, 
    dtype: torch.dtype, 
    device: torch.device
):
    config = MultiHeadAttentionConfig(
        n_heads = n_heads, 
        head_dim = head_dim, 
    )
    handlers = [
        TorchMultiHeadAttentionHandler(config), 
        FlashAttentionMultiHeadAttentionHandler(config), 
        FlashAttentionMutliHeadAttentionHandler2(config)
    ]
    hidden_size = n_heads * head_dim
    q = torch.randn(size=(batch_size, seq_len, hidden_size), dtype=dtype, device=device)
    k = torch.randn(size=(batch_size, seq_len, hidden_size), dtype=dtype, device=device)
    v = torch.randn(size=(batch_size, seq_len, hidden_size), dtype=dtype, device=device)
    params = MultiHeadAttentionParameters(return_scores=False)

    os = [handler(q, k, v, params).o for handler in handlers]
    o_ref = os[0]
    for i, o in enumerate(os):
        # print(f'o.view(-1)[:10]    : {o.view(-1)[:10]}')
        # print(f'o_ref.view(-1)[:10]: {o_ref.view(-1)[:10]}')
        assert o.shape == o_ref.shape
        assert torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__, '-s', '-x'])