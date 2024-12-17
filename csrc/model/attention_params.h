#pragma once
#include <torch/torch.h>
#include "memory/kv_cache.h"

namespace mllm{

class AttentionParameters{
public:
    KVCache* kv_cache;
    torch::Tensor q_cu_seq_lens;
    torch::Tensor k_cu_seq_lens;
    torch::Tensor paged_kv_last_page_len;
    torch::Tensor new_cache_slots;
    torch::Tensor block_tables;
    torch::Tensor cu_block_lens;
    int num_sequences;
    bool all_sequences_decode;
    int q_max_seq_len;
    int k_max_seq_len;
    // flash_infer_handler: "flashinfer.BatchPrefillWithPagedKVCacheWrapper" = None
};

}