import torch
from hydrainfer._C.kernel.reduce import sum
import pytest

@pytest.mark.parametrize("n_elements", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 333, 334])
@pytest.mark.parametrize("dtype", [torch.float, torch.half])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
def test_sum(
    n_elements: int, 
    dtype: torch.dtype, 
    device: torch.device
):
    input = torch.randn(n_elements, dtype=dtype, device=device)
    output = torch.zeros(1, dtype=dtype, device=device)
    sum(output, input)
    output_ref = torch.sum(input, dim=-1)

    assert torch.allclose(output, output_ref, atol=1e-2, rtol=1e-2), f'{output.item()}, {output_ref.item()}'

if __name__ == '__main__':
    pytest.main([__file__])