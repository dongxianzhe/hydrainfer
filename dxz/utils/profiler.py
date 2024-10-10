import time
import torch
from contextlib import contextmanager

@contextmanager
def profile(context_name: str, profile_time:bool=True, profile_nvtx_range:bool=True):
    if profile_nvtx_range:
        torch.cuda.nvtx.range_push(context_name)

    if profile_time:
        start_time = time.time()
    yield
    if profile_time:
        end_time = time.time()

    if profile_nvtx_range:
        torch.cuda.nvtx.range_pop()

    if profile_time:
        print(f'{context_name} took {end_time - start_time:.2}s')