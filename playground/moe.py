import torch
from torch import nn
from torch import Tensor
from dataclasses import dataclass
from hydrainfer.utils.logger import getLogger
logger = getLogger(__file__)

try:
    from hydrainfer._C.kernel import moe as moe_kernel
except ImportError as e:
    logger.warning("moe kernel import failed")
    moe_kernel = None


@dataclass
class MoEGateConfig:
    hidden_size: int
    n_experts: int
    topk: int
    score_func: str = "softmax"
    topk_method: str = "greedy"
    norm_topk_prob: bool = False
    routed_scaling_factor: float = 1.


class MoEGate(nn.Module):
    def __init__(self, config: MoEGateConfig):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(size=(config.n_experts, config.n_experts)))
        assert config.score_func == "softmax", f"not support {config.score_func}"
        assert config.topk_method == "greedy", f"not support {config.topk_method}"
        assert config.norm_topk_prob == False, f"not support {config.norm_topk_prob}"
        assert config.routed_scaling_factor == 1., f"not support {config.routed_scaling_factor}"

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        # hidden_states (n_tokens, hidden_size)
        # return topk_weights (n_tokens, topk) topk_idx (n_tokens, topk)
        n_tokens = hidden_states.shape[0]
        gating_logits = nn.functional.linear(hidden_states.to(torch.float32), self.weight.to(torch.float32), None)

        if moe_kernel is not None:
            topk_weights = hidden_states.new_empty(size=(n_tokens, self.config.topk))
            topk_indices = hidden_states.new_empty(size=(n_tokens, self.config.topk))
            moe_kernel.topk_softmax(gating_logits)
            return topk_weights, topk_indices

        if self.scoring_func == "softmax":
            scores = gating_logits.softmax(dim=-1, dtype=torch.float32) # (n_tokens, hidden_size)
        elif self.scoring_func == "sigmoid":
            scores = gating_logits.sigmoid() # (n_tokens, hidden_size)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        if self.topk_method == "greedy":
            topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, largest=True, sorted=False) # (n_tokens, top_k) (n_tokens, top_k)
        else:
            raise NotImplementedError(f"insupportable topk method for MoE gating: {self.topk_method}")

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator * self.routed_scaling_factor
        else:
            topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights, topk_indices