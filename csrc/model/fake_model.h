#pragma once
#include <torch/torch.h>
#include "model_params.h"
#include "linear.h"

namespace mllm{
struct FakeConfig{
    int hidden_size = 4096;
    int intermediate_size = 11008;
    int n_qo_heads = 32;
    int n_kv_heads = 32;
    int head_size = 128;
    int max_position_embeddings = 4096;
    float rope_theta = 10000;
    int n_layers = 32;
    float rms_norm_eps = 1e-5;
    int vocab_size = 32064;
    int image_token_id = 32000;
    int n_image_tokens = 576;
};

class FakeModelImpl : public torch::nn::Module{
public:
    int vocab_size_;
    int n_image_tokens_;
    int hidden_size_;
    torch::TensorOptions options_;
public:
    FakeModelImpl(const FakeConfig& config, const torch::TensorOptions& options) : vocab_size_(config.vocab_size), n_image_tokens_(config.n_image_tokens), hidden_size_(config.hidden_size), options_(options){
    }
    torch::Tensor forward(torch::Tensor input_ids, torch::Tensor pixel_values, torch::Tensor image_featues, torch::Tensor position_ids, const ModelParameters& model_params){
        int n_tokens = input_ids.size(0);
        return torch::randn({n_tokens, vocab_size_}, options_);
    }

    torch::Tensor image_embed(torch::Tensor pixel_values, const ModelParameters& model_params){
        int batch_size = pixel_values.size(0);
        auto image_features = torch::randn({batch_size, n_image_tokens_, hidden_size_}, options_);
        return image_features;
    }
};
TORCH_MODULE(FakeModel);

}