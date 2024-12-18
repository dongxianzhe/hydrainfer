#pragma once

#include<torch/torch.h>
#include<vector>

#include "block_allocator.h"

namespace mllm{
class KVCache final{
public:
    KVCache(int num_blocks, int block_size, int num_kv_heads, int head_size, const torch::TensorOptions& options);

    ~KVCache();

    KVCache(const KVCache&) = delete;
    KVCache(KVCache&&) = delete;
    KVCache& operator=(const KVCache&) = delete;
    KVCache& operator=(KVCache&&) = delete;

    std::vector<int> allocate(int n_blocks);
    
    void free(const std::vector<int>& blocks);
    
    std::tuple<torch::Tensor, torch::Tensor> get_kv_cache() const;

    void set_kv_cache(torch::Tensor slot_ids, torch::Tensor keys, torch::Tensor values);

private:
    int num_blocks_;
    int block_size_;
    int num_kv_heads_;
    int head_size_;
    torch::TensorOptions options_;
    torch::Tensor key_cache_;
    torch::Tensor value_cache_;
    BlockAllocator allocator_;
};
}