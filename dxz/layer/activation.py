import torch
from torch import nn, Tensor
import math
try:
    from dxz._C.kernel.activation import silu as silu_kernel
except ImportError:
    print('silu kernel import failed')
    silu_kernel = None

class NewGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)

def silu(h: Tensor):
    if silu_kernel is not None:
        return silu_kernel(h)
    return torch.nn.functional.silu(h)