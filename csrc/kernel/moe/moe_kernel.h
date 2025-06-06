#pragma once
#include <torch/torch.h>

namespace mllm::kernel{

void sum_out(const torch::Tensor& input,  // [n_tokens, topk, dim]
             torch::Tensor& output);       // [n_tokens, dim]
void grouped_topk_sigmoid(
    const torch::Tensor& gating_logits,    // [n_tokens, n_experts]
    const torch::Tensor& correction_bias,  // [n_experts]
    const int n_expert_groups,
    const int topk_group,
    const int topk,
    float scaling_factor,
    torch::Tensor& topk_weights,  // [n_tokens, topk]
    torch::Tensor& topk_indices   // [n_tokens, topk]
);
std::tuple<torch::Tensor, torch::Tensor> permute_with_index_map(
    torch::Tensor tokens,   // [n_tokens, dim]
    torch::Tensor topk_ids  // [n_tokens, topk]
);
torch::Tensor unpermute_with_index_map(
    torch::Tensor permuted_tokens,  // [n_permuted_tokens, dim]
    torch::Tensor row_id_map,       // [topk, n_tokens] => dst row
    torch::Tensor probs             // [n_tokens, topk]
);
std::tuple<torch::Tensor, torch::Tensor> permute_with_mask_map(
    torch::Tensor tokens,       // [n_tokens, dim]
    torch::Tensor routing_map,  // [n_tokens, n_experts] bool tensor
    int64_t topk);
torch::Tensor unpermute_with_mask_map(
    torch::Tensor permuted_tokens,  // [n_permuted_tokens, dim]
    torch::Tensor row_id_map,       // [n_experts, n_tokens] => dst row
    torch::Tensor probs             // [n_tokens, n_experts]
);
void topk_softmax(const torch::Tensor& gating_logits,  // [n_tokens, n_experts]
                  torch::Tensor& topk_weights,         // [n_tokens, topk]
                  torch::Tensor& topk_indices          // [n_tokens, topk]
);

}