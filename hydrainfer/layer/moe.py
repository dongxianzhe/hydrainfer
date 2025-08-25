import torch
from torch import nn, Tensor
from dataclasses import dataclass
from hydrainfer.utils.logger import getLogger
logger = getLogger(__file__)

try: 
    from hydrainfer._C.kernel.moe import topk_softmax as topk_softmax_kernel
except ImportError:
    logger.warning("import topk_softmax_kernel failed")
    topk_softmax_kernel = None

try:
    from hydrainfer._C.kernel.moe import permute_with_index_map as permute_with_index_map_kernel
except ImportError:
    logger.warning("import permute_with_index_map_kernel failed")
    permute_with_index_map_kernel = None

@dataclass
class MoeGateConfig:
    hidden_size: int
    n_routed_experts: int
    num_experts_per_tok: int

class MoEGate(nn.Module):
    def __init__(self, config: MoeGateConfig):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(size=(config.n_routed_experts, config.hidden_size)), requires_grad=False)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        # h (n_tokens, hidden_size)
        # return topk_weight (n_tokens, n_eperts_per_tok)
        # return topk_idx    (n_tokens, n_eperts_per_tok)
        n_tokens, hidden_size = h.shape
        logits = nn.functional.linear(h.type(torch.float32), self.weight.type(torch.float32), None) # (n_tokens, n_routed_experts)
        if topk_softmax_kernel is not None:
            topk_weight = torch.empty(size=(n_tokens, self.config.num_experts_per_tok), dtype=logits.dtype, device=logits.device)
            topk_idx = torch.empty(size=(n_tokens, self.config.num_experts_per_tok), dtype=logits.dtype, device=logits.device)
            topk_softmax_kernel(logits, topk_weight, topk_idx)
        else:
            scores = logits.softmax(dim=-1, dtype=torch.float32)
            topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False) # (n_tokens, n_experts_per_tok) (n_tokens, n_experts_per_tok)
            return topk_weight, topk_idx

@dataclass
class MoEConfig:
    hidden_size: int
    n_routed_experts: int
    num_experts_per_tok: int

class MoE(nn.Module):
    def __init__(self, config: MoeGateConfig, experts: nn.ModuleList):
        super().__init__()
        self.config = config
        self.gate = MoEGate(MoeGateConfig(
            hidden_size = config.hidden_size, 
            n_routed_experts = config.n_routed_experts, 
            num_experts_per_tok = config.num_experts_per_tok, 
        ))
        self.experts = experts

    def forward(self, h: Tensor) -> Tensor:
        # h (n_tokens, hidden_size)
        n_tokens, hidden_size = h.shape
        dtype, device = h.dtype, h.device

        topk_weight, topk_idx = self.gate(h) # (n_tokens, n_expert_per_token) (n_tokens, n_expert_per_token)
        if permute_with_index_map_kernel is not None:
            permuted_tokens, sorted_indices = permute_with_index_map_kernel(h, topk_idx) # (n_tokens * n_expert_per_token, hidden_size) with tokens with same expert grouped together
            raise Exception('todo')
        else:
            sorted_indices = topk_idx.view(-1).argsort()
            sorted_tokens = h[sorted_indices // self.config.num_experts_per_tok]

            cnts = torch.zeros(size=(n_tokens, self.config.n_routed_experts), dtype=dtype, device=device) # (n_tokens, n_experts)
            cnts.scatter_(dim=1, index=topk_idx, value=1) # (n_tokens, n_experts)
            tokens_per_expert = cnts.sum(dim=0, keepdim=False) # (n_experts, )
            tokens_per_expert.cpu().numpy()
            outputs: list[Tensor] = []
            start_idx: int = 0
            for i, (num_tokens, expert) in enumerate(zip(tokens_per_expert, self.experts)):
                end_idx = start_idx + num_tokens
                if num_tokens == 0:
                    continue
                tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
                expert_out = expert(tokens_for_this_expert)
                outputs.append(expert_out)
                start_idx = end_idx

            assert len(outputs) > 0
            o = torch.cat(outputs, dim=0)

            raise Exception('todo')