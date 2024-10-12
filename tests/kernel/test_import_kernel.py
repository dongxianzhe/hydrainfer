import torch

from dxz.kernel.gemm import sgemm_naive_cpu

a = torch.ones(size=(3, 3))
b = torch.ones(size=(3, 3))
c = torch.ones(size=(3, 3))
sgemm_naive_cpu(c, a, b)
print(a)
print(b)
print(c)

from dxz.kernel.flash_attn import mha_varlen_fwd
print(mha_varlen_fwd)