#pragma once
#include <cmath>
#include "flash_attn/flash_api.h"
namespace mllm{

class AttentionParams{
public:
    KVCache* kv_cache;
    torch::Tensor q_cu_seq_lens;
    torch::Tensor k_cu_seq_lens;
    // torch::Tensor paged_kv_last_page_len;
    torch::Tensor new_cache_slots;
    torch::Tensor block_tables;
    torch::Tensor cu_block_lens;
    int num_sequences;
    bool all_sequences_decode;
    int q_max_seq_len;
    int k_max_seq_len;
    // flash_infer_handler: "flashinfer.BatchPrefillWithPagedKVCacheWrapper" = None
};

class CausalGroupedQueryPageAttentionImpl : public torch::nn::Module{
public:
    int n_qo_heads_;
    int n_kv_heads_;
    int head_dim_;
    float sm_scale_;

    CausalGroupedQueryPageAttentionImpl(int n_qo_heads, int n_kv_heads, int head_dim) : n_qo_heads_(n_qo_heads), n_kv_heads_(n_kv_heads), head_dim_(head_dim), sm_scale_(1.0 / std::sqrt(head_dim)){
        CHECK(n_qo_heads % n_kv_heads == 0) << "n_qo_heads " << n_qo_heads << " not divisible by n_kv_heads " << n_kv_heads;
    }
    torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value, const AttentionParams& params){
        const int n_tokens = query.size(0);
        auto q = query.view({n_tokens, n_qo_heads_, head_dim_});
        auto k = key.view({n_tokens, n_kv_heads_, head_dim_});
        auto v = value.view({n_tokens, n_kv_heads_, head_dim_});
        params.kv_cache->set_kv_cache(params.new_cache_slots, k, v);
        auto out = torch::empty_like(q);
        auto [key_cache, value_cache] = params.kv_cache->get_kv_cache();
        std::cout << "sm_scale:" << sm_scale_ << std::endl;
        mha_varlen_fwd(
            out,
            q,
            key_cache, 
            value_cache, 
            params.q_cu_seq_lens, 
            params.k_cu_seq_lens, 
            params.block_tables, 
            params.cu_block_lens, 
            torch::nullopt, 
            params.q_max_seq_len, 
            params.k_max_seq_len, 
            sm_scale_, 
            0,
            -1,
            0,
            0
        );
        return out.view({-1, n_qo_heads_ * head_dim_});
    }
};
TORCH_MODULE(CausalGroupedQueryPageAttention);

}