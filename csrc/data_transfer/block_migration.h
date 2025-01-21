#pragma once

#include <torch/torch.h>
#include <vector>
#include <cuda_runtime.h>

namespace mllm::migration{

std::vector<int64_t> get_ipc_mem_handle(torch::Tensor tensor);

int64_t register_ipc_mem_handle(std::vector<int64_t> kv_cache_handle_vec);

void migrate_blocks(
	const int64_t prefill_start_head,
	const int64_t prefill_end_head,
	
	const std::vector<int64_t> &prefill_block_indexes,

	const int64_t decoding_start_head,
	const int64_t decoding_end_head,

	const std::vector<int64_t> &decoding_block_indexes,

	const int64_t prefill_dev_ptr_index,
	const int64_t num_heads,

	torch::Tensor decoding_worker_kv_cache
);

}