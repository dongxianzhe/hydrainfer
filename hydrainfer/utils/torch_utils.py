import torch

def str2device(device_type: str) -> torch.device:
    if device_type == 'cuda':
        return torch.device('cuda:0')
    if device_type == 'cuda:0':
        return torch.device('cuda:0')
    if device_type == 'cpu':
        return torch.device('cpu')
    raise Exception(f'device_type {device_type} is not supported')


def str2dtype(dtype: str) -> torch.dtype:
    if dtype == 'fp16':
        return torch.half
    if dtype == 'fp32':
        return torch.float
    raise Exception(f'dtype {dtype} is not supported')

def get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()