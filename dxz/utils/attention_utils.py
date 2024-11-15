import torch
from torch import Tensor

def sparsity(scores: Tensor, mask: Tensor, threshold: float) -> float:
    assert scores.dim() == 3, f"scores shape {scores.shape} should be (num_heads, seq_len, seq_len)"
    assert scores.shape == mask.shape, f"mask shape {mask.shape} should be same as scores shape {scores.shape}"
    assert mask.dtype == torch.bool, "mask.dtype should be bool"
    num_elements_gt_threshold = (scores > threshold).sum().item()
    num_unmasked_elements = mask.numel() - torch.sum(mask).item()
    return num_elements_gt_threshold / num_unmasked_elements