import torch
from torch import nn, Tensor
from hydrainfer.utils.logger import getLogger
logger = getLogger(__name__)

try:
    from hydrainfer._C.kernel.norm import rms_norm as rms_norm_kernel
except ImportError:
    logger.warning('rms_norm kernel import failed')
    rms_norm_kernel = None

def rmsnorm(h: Tensor, w: Tensor, eps: float) -> Tensor:
    if rms_norm_kernel is not None and h.device != torch.device('cpu'):
        o = torch.empty_like(h)
        rms_norm_kernel(o, h, w, eps)
        return o

    dtype = h.dtype
    rms = torch.sqrt(torch.mean(h.to(torch.float) ** 2, dim=-1, keepdim=True) + eps)
    normalized_x = h / rms
    o = normalized_x * w
    o = o.to(dtype)
    return o

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        output = rmsnorm(hidden_states, self.weight, self.variance_epsilon)
        return output