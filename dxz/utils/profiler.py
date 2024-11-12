import time
import torch
from contextlib import contextmanager
import atexit

context_total_times:dict[str, float] = {}
def print_profile_result():
    print('context_name   total_time')
    for context_name, total_time in context_total_times.items():
        print(f'{context_name:12}   {total_time:12} s')


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
        context_total_times[context_name] = context_total_times.get(context_name, 0) + end_time - start_time
        print(f'{context_name} took {end_time - start_time:.2}s')

atexit.register(print_profile_result)