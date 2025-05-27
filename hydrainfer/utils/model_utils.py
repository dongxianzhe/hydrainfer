import torch
from torch import nn

def count_parameters(model: nn.Module) -> tuple[int, int]:
    # return the number of parameters and parametesrrequired grad
    total = 0
    param_required_grad = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            param_required_grad += param.numel()
    return total, param_required_grad

def print_parameters(model: nn.Module) -> None:
    # print model parameters name and shape required grad
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name:30}: {param.shape}')