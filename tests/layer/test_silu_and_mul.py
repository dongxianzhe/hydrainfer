import torch
from torch import nn
from torch import Tensor
import pytest
from functools import partial
from hydrainfer.utils.benchmark_utils import benchmark
from hydrainfer.layer.activation import SiluAndMul

@pytest.mark.parametrize("n_tokens", [1, 2, 4, 8, 64, 128, 4096])
@pytest.mark.parametrize("hidden_size", [128, 1024, 4096, 8192,  11008])
@pytest.mark.parametrize("dtype", [torch.half])
@pytest.mark.parametrize("device", [torch.device('cuda:0'), torch.device('cpu')])
def test_silu_and_mul(
    n_tokens: int, 
    hidden_size: int, 
    dtype: torch.dtype, 
    device: torch.device, 
):
    x = torch.randn(size=(n_tokens, hidden_size * 2), dtype=dtype, device=device)
    def silu_and_mul_ref(x: Tensor):
        return nn.functional.silu(x[:, :hidden_size]) * x[:, hidden_size:]

    o_ref = silu_and_mul_ref(x)

    layers = [
        ("enable_flashinfer", SiluAndMul(enable_flashinfer=True)), 
        ("unable_flashinfer", SiluAndMul(enable_flashinfer=False)), 
    ]
    for name, layer in layers:
        o = layer(x)
        assert torch.allclose(o_ref, o, rtol=1e-2, atol=1e-2), f"{name} failed, max difference {torch.max(o, o_ref)}"
    print()
    print(f'n_tokens {n_tokens} hidden_size {hidden_size} dtype {dtype} device {device} ref layer latency {benchmark(partial(silu_and_mul_ref, x=x))} s')
    for name, layer in layers:
        print(f'n_tokens {n_tokens} hidden_size {hidden_size} dtype {dtype} device {device} {name} layer latency {benchmark(partial(layer, x=x))} s')

if __name__ == '__main__':
    pytest.main([__file__, '-s', '-x'])