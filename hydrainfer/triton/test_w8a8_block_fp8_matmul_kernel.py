import triton
import torch
import pytest
from hydrainfer.triton.w8a8_block_fp8_matmul_kernel import w8a8_block_fp8_matmul_forward

@pytest.mark.parametrize("block_size_n", [128])
@pytest.mark.parametrize("block_size_k", [128])
@pytest.mark.parametrize("M", [94])
@pytest.mark.parametrize("N", [24576])
@pytest.mark.parametrize("K", [1536])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
def test_w8a8_block_fp8_matmul_kernel(
    block_size_n: int, 
    block_size_k: int, 
    M: int, 
    N: int, 
    K: int, 
    device: torch.device, 
):
    block_n_num = triton.cdiv(N, block_size_n)
    block_k_num = triton.cdiv(K, block_size_k)

    A = torch.randn(size=(M, K), device=device).to(torch.float8_e4m3fn)
    B = torch.randn(size=(N, K), device=device).to(torch.float8_e4m3fn)
    As = torch.randint(20, 30, size=(M, block_k_num), dtype=torch.float, device=device)
    Bs = torch.randint(20, 30, size=(block_n_num, block_k_num), dtype=torch.float, device=device)
    block_size = [block_size_n, block_size_k]

    config = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": block_size[0],
        "BLOCK_SIZE_K": block_size[1],
        "GROUP_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 2,
    }

    C, shared_mem_bytes = w8a8_block_fp8_matmul_forward(A, B, As, Bs, block_size, config, torch.half)

    A_ref = A.to(torch.float)
    B_ref = B.to(torch.float)
    As_ref = As.to(torch.float).repeat_interleave(block_size_k, dim=1)
    Bs_ref = Bs.to(torch.float).repeat_interleave(block_size_n, dim=0).repeat_interleave(block_size_k, dim=1)

    C_ref = torch.matmul(A_ref * As_ref, (B_ref * Bs_ref).transpose(0, 1)).to(C.dtype)
    print(C)
    print(C_ref)
    assert torch.allclose(C, C_ref, atol=1, rtol=0.1)

if __name__ == '__main__':
    pytest.main([__file__, '-s'])