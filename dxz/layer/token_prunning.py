import torch
from torch import nn, Tensor
import math

def focal_prunning(tokens: Tensor, scores: Tensor, n_output_tokens: int, strategy: str):
    # tokens (batch_size, n_tokens, hidden_size)
    # scores (batch_size, n_heads, n_tokens, n_tokens)
    assert tokens.dim() == 3
    batch_size, n_tokens, hidden_size = tokens.shape
    assert scores.dim() == 4
    assert scores.shape[2] == n_tokens
    assert scores.shape[3] == n_tokens
    assert scores.shape[0] == batch_size
    dtype = tokens.dtype
    device= tokens.device

    scores = torch.mean(scores, dim=1, keepdim=False) # (batch_size, n_tokens, n_tokens)
    token_pruned = torch.zeros(size=(batch_size, n_output_tokens, hidden_size), dtype=dtype, device=device)
    for i in range(batch_size):
        s1 = torch.mean(scores[i], dim=0, keepdim=False)
        s2 = torch.mean(scores[i], dim=1, keepdim=False)
        sig = s1 if torch.var(s1).item() > torch.var(s2).item() else s2

        if strategy == 'rank':
            _, sort_idx = torch.sort(sig, descending=True)
            selected_ids = sort_idx[:n_output_tokens]
        else:
            n_rows = math.sqrt(n_tokens)
            assert n_rows.is_integer(), f'{n_tokens} is not a square number'
            n_rows = int(n_rows)
            sig = torch.sum(sig.reshape(n_rows, n_rows), dim=-1, keepdim=False)
            _, sort_idx = torch.sort(sig, descending=True)
            sort_idx = sort_idx[: n_output_tokens]
            selected_ids = torch.arange(n_tokens, dtype=torch.int, device=tokens.device).reshape(n_rows, n_rows)[sort_idx, :].reshape(-1)
        selected_ids, _ = torch.sort(selected_ids)
        token_pruned[i, :, :] = tokens[i, selected_ids, :]
    return token_pruned

# deprecate
class FocalPrunning(nn.Module):
    def __init__(self, ratio: float, strategy: str):
        super().__init__()
        assert strategy in ['rank', 'row']
        self.strategy = strategy
        self.ratio = ratio

    def forward(self, tokens: Tensor, scores: Tensor):
        # tokens (..., n_tokens, hidden_size)
        # scores (n_heads, n_tokens, n_tokens)
        assert scores.dim() == 3
        assert scores.shape[1] == scores.shape[2]
        n_heads, n_tokens, _ = scores.shape
        scores = torch.mean(scores, dim=0, keepdim=False)
        s1 = torch.mean(scores, dim=1, keepdim=False)
        s2 = torch.mean(scores, dim=0, keepdim=False)
        sig = s1 if torch.var(s1).item() > torch.var(s2).item() else s2

        if self.strategy == 'rank':
            _, sort_idx = torch.sort(sig, descending=True)
            selected_ids = sort_idx[:math.floor(n_tokens * self.ratio)]
        else:
            n_rows = math.sqrt(n_tokens)
            assert n_rows.is_integer(), f'{n_tokens} is not a square number'
            n_rows = int(n_rows)
            sig = torch.sum(sig.reshape(n_rows, n_rows), dim=-1, keepdim=False)
            _, sort_idx = torch.sort(sig, descending=True)
            sort_idx = sort_idx[: math.floor(n_rows * self.ratio)]
            selected_ids = torch.arange(n_tokens, dtype=torch.int, device=tokens.device).reshape(n_rows, n_rows)[sort_idx, :].reshape(-1)
        
        selected_ids, _ = torch.sort(selected_ids)
        token_pruned = tokens[..., selected_ids, :]
        return token_pruned
