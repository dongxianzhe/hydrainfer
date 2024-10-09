import torch
from torch import nn, Tensor
from transformers import MixtralConfig
from torch.nn import functional

class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts)
        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(config.num_local_experts)])
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states (n_tokens, hidden_state)
        router_logits = self.gate(hidden_states) # (n_tokens, num_local_experts)
        routing_weights = functional.softmax(router_logits, dim=-1, dtype=torch.float) # (n_tokens, num_local_experts)
        routing_weights, selected_experts = torch.topk(routing_weights, self.config.num_experts_per_tok, dim=-1)
        routing_weights = routing_weights.to(hidden_states.dtype)
        # routing_weights(n_tokens, num_experts_per_tok) selected_experts(n_tokens, num_experts_per_tok)
        expert_mask = functional.one_hot(selected_experts, num_classes=self.config.num_local_experts).permute(2, 1, 0)
        # expert_mask (n_tokens, num_experts_per_tok, num_local_experts) 
        # -- permute -> (num_local_experts, num_experts_per_tok, n_tokens)  

        final_hidden_states = torch.zeros(size=(hidden_states.shape[0], self.config.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        for expert_idx in range(self.config.num_local_experts):
            # exper_mask[expert_idx] (num_experts_per_tok, n_tokens)
            weight_ids, token_ids = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_ids]
            # current_state (n, hidden_size)
            current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[token_ids, weight_ids, None]
            # current_hidden_states (n, hidden_size)
            final_hidden_states.index_add_(dim=0, index=token_ids, source=current_hidden_states.to(hidden_states.dtype))

        return final_hidden_states

if __name__ == '__main__':
    config = MixtralConfig()
    model = MixtralSparseMoeBlock(config)