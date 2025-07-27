import triton
import pytest
import torch

from hydrainfer.triton.per_token_group_quant_fp8_kernel import _per_token_group_quant_fp8

@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("M", [2560])
@pytest.mark.parametrize("N", [2048])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
@pytest.mark.parametrize("column_major_scales", [False, True])
def test_per_token_group_quant_fp8_kernel(
    group_size: int, 
    M: int, 
    N: int, 
    column_major_scales: bool, 
    device: torch.device, 
):
    x = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    x_q, x_s, shared_mem_bytes = _per_token_group_quant_fp8(x, group_size, column_major_scales=column_major_scales)
    x_ref = x_q.to(torch.float) * x_s.repeat_interleave(group_size, dim=-1)

    assert torch.allclose(x.to(torch.float), x_ref, atol=0.1, rtol=0.1)
    assert x_s.shape == (M, N // group_size)
    if column_major_scales:
        assert x_s.stride() == (1, M)
    else:
        assert x_s.stride() == (N // group_size, 1)

if __name__ == '__main__':
    pytest.main([__file__, '-s'])