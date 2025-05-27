import torch
from hydrainfer._C.kernel.activation import silu
from torch.nn.functional import silu as silu_torch
import pytest

@pytest.mark.parametrize("n_tokens", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("hidden_size", [1, 2, 32, 1024, 4096, 333, 334])
@pytest.mark.parametrize("dtype", [torch.float, torch.half])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
def test_silu(
    n_tokens: int, 
    hidden_size: int, 
    dtype: torch.dtype, 
    device: torch.device
):
    input = torch.randn((n_tokens, hidden_size), dtype=dtype, device=device)
    output_ref = silu_torch(input)
    output = silu(input)
    assert torch.allclose(output, output_ref, atol=1e-3, rtol=1e-3), f'{output.view(-1)[:10]}, {output_ref.view(-1)[:10]}'

if __name__ == '__main__':
    pytest.main([__file__])