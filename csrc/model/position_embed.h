#pragma once

#include <torch/torch.h>
#include <c10/core/TensorOptions.h>
#include "position_embedding/rope.h"

namespace mllm{

torch::Tensor compute_default_inv_freq(int64_t rotary_dim, float theta) {
    CHECK(rotary_dim % 2 == 0) << "rotary_dim must be even";
    const auto slice = torch::arange(0, rotary_dim, 2, torch::kFloat32);
    return 1.0 / torch::pow(theta, slice / rotary_dim);
}

class RotaryEmbeddingImpl : public torch::nn::Module{
public:
    int rotary_dim_;
    int max_position_embeddings_;
    bool interleaved_;
    torch::Tensor cos_sin_cache_;

    RotaryEmbeddingImpl(int rotary_dim, int max_position_embeddings, torch::Tensor inv_freq, bool interleaved, const torch::TensorOptions& options) : rotary_dim_(rotary_dim), max_position_embeddings_(max_position_embeddings), interleaved_(interleaved) {
        auto t = torch::arange(max_position_embeddings, torch::dtype(torch::kFloat32));
        auto freqs = torch::einsum("i,j->ij", {t, inv_freq});
        auto cos = freqs.cos().index({torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice()});
        auto sin = freqs.sin().index({torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice()});
        auto cos_sin = torch::cat({cos, sin}, 1);
        cos_sin_cache_ =register_buffer("cos_sin_cache_", cos_sin.to(options)); 
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& position_ids){
        torch::Tensor _query = query;
        torch::Tensor _key = key;
        kernel::apply_rotary_pos_emb(_query, _key, position_ids, cos_sin_cache_, rotary_dim_, interleaved_);
        return std::make_tuple(query, key);
    }
};

TORCH_MODULE(RotaryEmbedding);


}