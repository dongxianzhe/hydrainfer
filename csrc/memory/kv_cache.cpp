#include "kv_cache.h"

#include <glog/logging.h>

#include "kv_cache_kernels/kv_cache_kernels.h"

namespace mllm{

KVCache::KVCache(int num_blocks, int block_size, int num_kv_heads, int head_size, torch::Dtype dtype, torch::Device device) : num_blocks_(num_blocks), block_size_(block_size), num_kv_heads_(num_kv_heads), head_size_(head_size), dtype_(dtype), device_(device), allocator_(BlockAllocator(num_blocks)){
    key_cache_ = torch::randn({num_blocks, block_size, num_kv_heads, head_size}, torch::dtype(dtype_).device(device_));
    value_cache_ = torch::randn({num_blocks, block_size, num_kv_heads, head_size}, torch::dtype(dtype_).device(device_));
}

KVCache::~KVCache() {}

std::vector<int> KVCache::allocate(int n_blocks){
    return allocator_.allocate(n_blocks);
}

void KVCache::free(const std::vector<int> blocks){
    allocator_.free(blocks);
}

std::tuple<torch::Tensor, torch::Tensor> KVCache::get_kv_cache() const{
    return {key_cache_, value_cache_};
}

void KVCache::set_kv_cache(torch::Tensor slot_ids, torch::Tensor keys, torch::Tensor values){
    DCHECK_EQ(slot_ids.size(0), keys.size(0));
    DCHECK_EQ(slot_ids.size(0), values.size(0));
    DCHECK_EQ(slot_ids.device(), keys.device());
    DCHECK_EQ(slot_ids.device(), values.device());
    kernel::set_kv_cache(slot_ids, keys, values, key_cache_, value_cache_);
}

}