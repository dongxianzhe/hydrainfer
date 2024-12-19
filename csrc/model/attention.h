#pragma once
#include <cmath>
#include "memory/kv_cache.h"
#include "flash_attn/flash_api.h"
#include "attention_params.h"
namespace mllm{

class MultiHeadAttentionImpl : public torch::nn::Module{
public:
    int n_heads_;
    int head_size_;
    torch::TensorOptions options_;
public:
    MultiHeadAttentionImpl(int n_heads, int head_size, const torch::TensorOptions& options) : n_heads_(n_heads), head_size_(head_size), options_(options){
    }
    torch::Tensor forward(torch::Tensor query, torch::Tensor key, torch::Tensor value){
        int batch_size  = query.size(0);
        int seq_len     = query.size(1);
        int hidden_size = query.size(2);

        query = query.view({batch_size * seq_len, n_heads_, head_size_});
        key   = key.view(  {batch_size * seq_len, n_heads_, head_size_});
        value = value.view({batch_size * seq_len, n_heads_, head_size_});
        
        auto o = torch::empty({batch_size * seq_len, n_heads_, head_size_}, options_);
        auto q_cu_seq_lens = torch::arange(0, (batch_size + 1) * seq_len, seq_len, options_.dtype(torch::kInt32));
        auto k_cu_seq_lens = torch::arange(0, (batch_size + 1) * seq_len, seq_len, options_.dtype(torch::kInt32));
        mha_varlen_fwd(
            o, 
            query, 
            key, 
            value, 
            q_cu_seq_lens, 
            k_cu_seq_lens,
            torch::nullopt, 
            torch::nullopt, 
            torch::nullopt, 
            seq_len, 
            seq_len, 
            1. / std::sqrt(head_size_),
            0, 
            -1,
            -1, 
            0
        );
        return o.view({batch_size, seq_len, hidden_size});
    }
};
TORCH_MODULE(MultiHeadAttention);


class CausalGroupedQueryPageAttentionImpl : public torch::nn::Module{
public:
    int n_qo_heads_;
    int n_kv_heads_;
    int head_dim_;
    float sm_scale_;

    CausalGroupedQueryPageAttentionImpl(int n_qo_heads, int n_kv_heads, int head_dim) : n_qo_heads_(n_qo_heads), n_kv_heads_(n_kv_heads), head_dim_(head_dim), sm_scale_(1.0 / std::sqrt(head_dim)){
        CHECK(n_qo_heads % n_kv_heads == 0) << "n_qo_heads " << n_qo_heads << " not divisible by n_kv_heads " << n_kv_heads;
    }
    torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value, const AttentionParameters& params){
        const int n_tokens = query.size(0);
        auto q = query.view({n_tokens, n_qo_heads_, head_dim_});
        auto k = key.view({n_tokens, n_kv_heads_, head_dim_});
        auto v = value.view({n_tokens, n_kv_heads_, head_dim_});
        params.kv_cache->set_kv_cache(params.new_cache_slots, k, v);
        auto out = torch::empty_like(q);
        auto [key_cache, value_cache] = params.kv_cache->get_kv_cache();
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