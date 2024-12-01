import torch
from torch import nn, Tensor
import math

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
