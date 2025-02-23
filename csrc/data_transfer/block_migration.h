#pragma once

#include <torch/torch.h>
#include <vector>
#include <cuda_runtime.h>

namespace mllm::migration{

std::vector<int64_t> get_ipc_mem_handle(torch::Tensor tensor);

int64_t register_ipc_mem_handle(std::vector<int64_t> kv_cache_handle_vec);

void migrate_blocks(
	const std::vector<int64_t>& src_block_table, 
	const std::vector<int64_t>& dst_block_table, 
	const std::vector<int64_t>& src_cache, // (n_layers, n_tokens, n_blocks, block_size, n_heads, head_size)
	torch::Tensor dst_cache               // (n_layers, n_tokens, n_blocks, block_size, n_heads, head_size)
);

}