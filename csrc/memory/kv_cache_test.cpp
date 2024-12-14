#include "kv_cache.h"
#include <iostream>

#include <torch/torch.h>

#include <gtest/gtest.h>

namespace mllm{
TEST(KVCacheTest, creation){
    int num_blocks = 10000;
    int block_size = 16;
    int num_kv_heads = 32;
    int head_size = 128;
    torch::Dtype dtype(torch::kHalf);
    torch::Device device(torch::kCUDA);

    KVCache kv_cache(num_blocks, block_size, num_kv_heads, head_size, dtype, device);
}

TEST(KVCacheTest, set_kv_cache){
    int num_blocks = 10000;
    int block_size = 16;
    int num_kv_heads = 32;
    int head_size = 128;
    torch::Dtype dtype(torch::kHalf);
    torch::Device device(torch::kCUDA);
    int n_tokens = 100;

    KVCache kv_cache(num_blocks, block_size, num_kv_heads, head_size, dtype, device);
    auto [key_cache, value_cache] = kv_cache.get_kv_cache();
    torch::Tensor key_cache_ref = key_cache.clone();
    torch::Tensor value_cache_ref = value_cache.clone();

    torch::Tensor slot_ids = torch::randperm(num_blocks, torch::dtype(torch::kInt).device(device)).slice(0, 0, n_tokens);
    torch::Tensor key = torch::randn({n_tokens, num_kv_heads, head_size}, torch::dtype(dtype).device(device));
    torch::Tensor value = torch::randn({n_tokens, num_kv_heads, head_size}, torch::dtype(dtype).device(device));

    EXPECT_TRUE(torch::equal(key_cache, key_cache_ref));
    EXPECT_TRUE(torch::equal(value_cache, value_cache_ref));

    using ISlice = torch::indexing::Slice;
    for(int i = 0;i < n_tokens;i ++){
        const int slot_id = slot_ids[i].item<int>();
        const int block_id = slot_id / block_size;
        const int block_offset = slot_id % block_size;
        key_cache_ref.index_put_({block_id, block_offset, ISlice(), ISlice()}, key[i]);
        value_cache_ref.index_put_({block_id, block_offset, ISlice(), ISlice()}, value[i]);
    }

    EXPECT_FALSE(torch::equal(key_cache, key_cache_ref));
    EXPECT_FALSE(torch::equal(value_cache, value_cache_ref));

    kv_cache.set_kv_cache(slot_ids, key, value);

    EXPECT_TRUE(torch::equal(key_cache, key_cache_ref));
    EXPECT_TRUE(torch::equal(value_cache, value_cache_ref));
}

}