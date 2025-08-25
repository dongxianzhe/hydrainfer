import torch

M, N, K = 94, 24576, 1536

A = torch.full((M, K), 1.5, dtype=torch.float8_e4m3fn, device='cuda')
B = torch.full((N, K), 1.5, dtype=torch.float8_e4m3fn, device='cuda')