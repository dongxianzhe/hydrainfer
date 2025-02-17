#pragma once
#include <torch/torch.h>

namespace mllm::kernel{
void set_image_cache(
    const torch::Tensor& slot_ids,          // [n_tokens]
    const torch::Tensor& image_tokens,      // [n_tokens, n_heads, head_dim]
    torch::Tensor& image_cache);            // [n_blocks, block_size, n_heads, head_dim]
}