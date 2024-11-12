import torch
from torch import Tensor

def rms_norm(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float
    ):
    ...