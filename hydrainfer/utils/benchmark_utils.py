import time
import torch

@torch.inference_mode
def benchmark(forward_fn, warmup=3, n_iter=100):
    for _ in range(warmup):
        forward_fn()
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iter):
        forward_fn()
        torch.cuda.synchronize()
    end = time.time()

    latency = (end - start) / n_iter
    return latency