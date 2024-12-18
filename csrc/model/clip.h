#pragma once
#include <torch/torch.h>
#include "linear.h"
#include <cmath>
#include "conv.h"
#include "embedding.h"
#include "attention.h"

namespace mllm{

class CLIPConfig{
public:
    int hidden_size = 1024;
    int intermediate_size = 4096;
    int head_size = 64;
    int n_heads = 16;
    int n_layers = 24;
    int num_channels = 3;
    int patch_size = 14;
    int image_size = 336;
    float layer_norm_eps = 1e-5;
    int image_token_index = 32000;
};

class CLIPSdpaAttentionImpl : public torch::nn::Module{
public:
    Linear q_proj_{nullptr};
    Linear k_proj_{nullptr};
    Linear v_proj_{nullptr};
    Linear out_proj_{nullptr};
    MultiHeadAttention attention_{nullptr};
    int n_heads_;
    int hidden_size_;
    int head_size_;
public:
    CLIPSdpaAttentionImpl(const CLIPConfig& config, const torch::TensorOptions& options) : n_heads_(config.n_heads), hidden_size_(config.hidden_size), head_size_(config.head_size){
        q_proj_ = register_module("q_proj", Linear(hidden_size_, hidden_size_, true, options));
        k_proj_ = register_module("k_proj", Linear(hidden_size_, hidden_size_, true, options));
        v_proj_ = register_module("v_proj", Linear(hidden_size_, hidden_size_, true, options));
        out_proj_ = register_module("out_proj", Linear(hidden_size_, hidden_size_, true, options));
        attention_ = register_module("attention", MultiHeadAttention(n_heads_, head_size_, options));
    }
    torch::Tensor forward(torch::Tensor h){
        auto query = q_proj_(h);
        auto key   = k_proj_(h);
        auto value = v_proj_(h);
        h = attention_(query, key, value);
        auto o = out_proj_(h);
        return o;
    }
};
TORCH_MODULE(CLIPSdpaAttention);

class QuickGELUActivationImpl : public torch::nn::Module{
public:
    QuickGELUActivationImpl(){}
    torch::Tensor forward(torch::Tensor x){
        return x * torch::sigmoid(1.702 * x);
    }
};
TORCH_MODULE(QuickGELUActivation);

class CLIPMLPImpl : public torch::nn::Module{
public:
    Linear fc1_{nullptr};
    Linear fc2_{nullptr};
    QuickGELUActivation activation_fn_{nullptr};
public:
    CLIPMLPImpl(const CLIPConfig& config, const torch::TensorOptions& options){
        fc1_ = register_module("fc1", Linear(config.hidden_size, config.intermediate_size, true, options));
        fc2_ = register_module("fc2", Linear(config.intermediate_size, config.hidden_size, true, options));
        activation_fn_ = register_module("activation_fn_", QuickGELUActivation());
    }
    torch::Tensor forward(torch::Tensor h){
        h = fc1_(h);
        h = activation_fn_(h);
        h = fc2_(h);
        return h;
    }
};
TORCH_MODULE(CLIPMLP);

class LayerNormImpl: public torch::nn::Module{
public:
    int hidden_size_;
    float eps_;
    torch::Tensor weight_{nullptr};
    torch::Tensor bias_{nullptr};
public:
    LayerNormImpl(int hidden_size, float eps, bool bias, const torch::TensorOptions& options) : hidden_size_(hidden_size), eps_(eps){
        weight_ = register_parameter("weight", torch::empty({hidden_size_}, options), false);
        if(bias) bias_ = register_parameter("bias", torch::empty({hidden_size_}, options), false);
    }
    torch::Tensor forward(torch::Tensor x){
        return torch::nn::functional::detail::layer_norm(x, {hidden_size_}, weight_, bias_, eps_);
    }
};
TORCH_MODULE(LayerNorm);

class CLIPEncoderLayerImpl : public torch::nn::Module{
public:
    CLIPSdpaAttention self_attn_{nullptr};
    LayerNorm layer_norm1_{nullptr};
    CLIPMLP mlp_{nullptr};
    LayerNorm layer_norm2_{nullptr};
    int layer_id_;
public:
    CLIPEncoderLayerImpl(const CLIPConfig& config, const torch::TensorOptions& options, int layer_id) : layer_id_(layer_id){
        self_attn_ = register_module("self_attn", CLIPSdpaAttention(config, options));
        layer_norm1_ = register_module("layer_norm1", LayerNorm(config.hidden_size, config.layer_norm_eps, true, options));
        mlp_ = register_module("mlp", CLIPMLP(config, options));
        layer_norm2_ = register_module("layer_norm2", LayerNorm(config.hidden_size, config.layer_norm_eps, true, options));
    }
    torch::Tensor forward(torch::Tensor h){
        auto r = h;
        h = layer_norm1_(h);
        h = self_attn_(h);
        h = r + h;
        r = h;
        h = layer_norm2_(h);
        h = mlp_(h);
        h = r + h;
        return h;
    }
};
TORCH_MODULE(CLIPEncoderLayer);

class CLIPVisionEmbeddingsImpl : public torch::nn::Module{
public:
    int hidden_size_;
    int image_size_;
    int patch_size_;
    int num_channels_;
    int num_positions_;
    
    torch::Tensor class_embedding_{nullptr};
    Conv2d patch_embedding_{nullptr};
    Embedding position_embedding_{nullptr};
    torch::Tensor position_ids_{nullptr};
public:
    CLIPVisionEmbeddingsImpl(CLIPConfig config, const torch::TensorOptions& options) : hidden_size_(config.hidden_size), image_size_(config.image_size), patch_size_(config.patch_size), num_channels_(config.num_channels), num_positions_((config.image_size / config.patch_size) * (config.image_size / config.patch_size) + 1){
        class_embedding_ = register_parameter("class_embedding", torch::empty({hidden_size_}, options), false);
        patch_embedding_ = register_module("patch_embedding", Conv2d(config.num_channels, config.hidden_size, config.patch_size, config.patch_size, false, options));
        position_embedding_ = register_module("position_embedding", Embedding(num_positions_, config.hidden_size, options));
        auto p = torch::arange(num_positions_, options.dtype(torch::kInt));
        position_ids_ = register_buffer("position_ids", p); 
    }
    torch::Tensor forward(torch::Tensor pixel_values){
        int batch_size = pixel_values.size(0);
        auto patch_embeds = patch_embedding_(pixel_values.to(patch_embedding_->weight_.dtype()));
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2);
        auto class_embeds = class_embedding_.index({torch::indexing::None, torch::indexing::None, torch::indexing::Slice()});
        auto embeddings = torch::cat({class_embeds, patch_embeds}, 1);
        embeddings = embeddings + position_embedding_(position_ids_.unsqueeze(0));
        return embeddings;
    }
};
TORCH_MODULE(CLIPVisionEmbeddings);


}