import torch
from torch import nn
from hydrainfer._C.kernel.norm import rms_norm
import pytest

@pytest.mark.parametrize("n_tokens", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("hidden_size", [1, 2, 32, 1024, 4096, 333, 334])
@pytest.mark.parametrize("epsilon", [1e-5])
@pytest.mark.parametrize("dtype", [torch.float, torch.half])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
def test_rms_norm(
    n_tokens: int, 
    hidden_size: int, 
    epsilon: float,
    dtype: torch.dtype, 
    device: torch.device
):
    model = nn.RMSNorm(normalized_shape=(hidden_size, ), eps=epsilon)
    model.to(dtype=dtype, device=device)
    input = torch.randn((n_tokens, hidden_size), dtype=dtype, device=device)
    output = torch.zeros((n_tokens, hidden_size), dtype=dtype, device=device)
    output_ref = model(input)
    rms_norm(output, input, model.weight, epsilon)
    assert torch.allclose(output, output_ref, atol=1e-3, rtol=1e-3), f'{output.item()}, {output_ref.item()}'
    assert torch.allclose(output, output_ref, atol=1e-3, rtol=1e-3), f'{output.item()}, {output_ref.item()}'

if __name__ == '__main__':
    pytest.main([__file__])