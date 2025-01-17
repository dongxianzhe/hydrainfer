import torch
from torch import nn, Tensor

try:
    from dxz._C.kernel.norm import rms_norm as rms_norm_kernel
except ImportError:
    print('rms_norm kernel import failed')
    rms_norm_kernel = None

def rmsnorm(h: Tensor, w: Tensor, eps: float) -> Tensor:
    if rms_norm_kernel is not None:
        o = torch.empty_like(h)
        rms_norm_kernel(o, h, w, eps)
        return o

    dtype = h.dtype
    rms = torch.sqrt(torch.mean(h.to(torch.float) ** 2, dim=-1, keepdim=True) + eps)
    print(rms.shape)
    normalized_x = h / rms
    o = normalized_x * w
    o = o.to(dtype)
    return o