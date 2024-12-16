#pragma once

#include <torch/torch.h>
#include "norm/rms_norm.h"
#include "memory/kv_cache.h"
#include "attention.h"
#include "position_embed.h"
#include "linear.h"
#include "embedding.h"

namespace mllm{

class ModelParameters{
public:
    std::vector<AttentionParams*> attention_params;
};

class LlamaConfig{
public:
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
};


class LlamaMLPImpl : public torch::nn::Module{
public:
    Linear gate_proj_{nullptr};
    Linear up_proj_{nullptr};
    Linear down_proj_{nullptr};
    LlamaMLPImpl(const LlamaConfig& config, const torch::TensorOptions& options){
        gate_proj_ = register_module("gate_proj", Linear(config.hidden_size      , config.intermediate_size, false, options));
        up_proj_   = register_module("up_proj"  , Linear(config.hidden_size      , config.intermediate_size, false, options));
        down_proj_ = register_module("down_proj", Linear(config.intermediate_size, config.hidden_size      , false, options));
    }
    torch::Tensor forward(const torch::Tensor& hidden_states){
        return down_proj_(torch::nn::functional::silu(gate_proj_(hidden_states)) * up_proj_(hidden_states));
    }
};
TORCH_MODULE(LlamaMLP);

class LlamaRMSNormImpl : public torch::nn::Module{
public:
    int hidden_size_;
    float rms_norm_eps_;
    torch::Tensor weight_;
    LlamaRMSNormImpl(int hidden_size, float rms_norm_eps, const torch::TensorOptions& options) : hidden_size_(hidden_size), rms_norm_eps_(rms_norm_eps){
        weight_ = register_parameter("weight", torch::ones({hidden_size}, options), false);
    }
    torch::Tensor forward(torch::Tensor hidden_states){    
        auto out = torch::empty_like(hidden_states);
        kernel::rms_norm(out, hidden_states, weight_, rms_norm_eps_);
        return out;
    }
};
TORCH_MODULE(LlamaRMSNorm);

class LlamaSdpaAttentionImpl : public torch::nn::Module{
public:
    Linear q_proj_{nullptr};
    Linear k_proj_{nullptr};
    Linear v_proj_{nullptr};
    Linear o_proj_{nullptr};
    RotaryEmbedding rotary_embed_{nullptr};
    CausalGroupedQueryPageAttention attention_{nullptr};
    int n_qo_heads_;
    int n_kv_heads_;
    int head_size_;
    int hidden_size_;
    LlamaSdpaAttentionImpl(const LlamaConfig& config, const torch::TensorOptions& options) : n_qo_heads_(config.n_qo_heads), n_kv_heads_(config.n_kv_heads), head_size_(config.head_size), hidden_size_(config.hidden_size){
        CHECK(n_qo_heads_ * head_size_ == hidden_size_);
        q_proj_ = register_module("q_proj", Linear(hidden_size_, hidden_size_, false, options));
        k_proj_ = register_module("k_proj", Linear(hidden_size_, n_kv_heads_* head_size_, false, options));
        v_proj_ = register_module("v_proj", Linear(hidden_size_, n_kv_heads_* head_size_, false, options));
        o_proj_ = register_module("o_proj", Linear(hidden_size_, hidden_size_, false, options));
        rotary_embed_ = register_module("rotary_emb", RotaryEmbedding(
            config.head_size, 
            config.max_position_embeddings, 
            compute_default_inv_freq(config.head_size, config.rope_theta), 
            false,
            options
        ));
        attention_ = register_module("attention", CausalGroupedQueryPageAttention(
            config.n_qo_heads, 
            config.n_kv_heads, 
            config.head_size
        ));
    }
    torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor position_ids, const AttentionParams& attention_params){
        auto query = q_proj_(hidden_states);
        auto key   = k_proj_(hidden_states);
        auto value = v_proj_(hidden_states);
        query = query.view({-1, n_qo_heads_, head_size_});
        key   = key.view({-1, n_kv_heads_, head_size_});
        value = value.view({-1, n_kv_heads_, head_size_});
        auto [query_rot, key_rot] = rotary_embed_->forward(query, key, position_ids);
        hidden_states = attention_(query_rot, key_rot, value, attention_params);
        return o_proj_(hidden_states);
    }
};
TORCH_MODULE(LlamaSdpaAttention);



class LlamaDecoderLayerImpl : public torch::nn::Module{
public:
    int layer_id_;
    LlamaSdpaAttention self_attn_{nullptr};
    LlamaMLP mlp_{nullptr};
    LlamaRMSNorm input_layernorm_{nullptr};
    LlamaRMSNorm post_attention_layernorm_{nullptr};
public:
    LlamaDecoderLayerImpl(const LlamaConfig& config, const torch::TensorOptions& options, int layer_id) : layer_id_(layer_id){
        self_attn_ = register_module("self_attn", LlamaSdpaAttention(config, options));
        mlp_       = register_module("mlp"      , LlamaMLP(config, options));
        input_layernorm_ = register_module("input_layernorm", LlamaRMSNorm(config.hidden_size, config.rms_norm_eps, options));
        post_attention_layernorm_ = register_module("post_attention_layernorm", LlamaRMSNorm(config.hidden_size, config.rms_norm_eps, options));
    }
    torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor position_ids, const ModelParameters& model_params){
        auto residual = hidden_states;
        hidden_states = input_layernorm_(hidden_states);
        hidden_states = self_attn_(hidden_states, position_ids, *model_params.attention_params[layer_id_]);
        hidden_states = residual + hidden_states;
        residual = hidden_states;
        hidden_states = post_attention_layernorm_(hidden_states);
        hidden_states = mlp_(hidden_states);
        hidden_states = residual + hidden_states;
        return hidden_states;
    }
};
TORCH_MODULE(LlamaDecoderLayer);

class LlamaModelImpl : public torch::nn::Module{
public:
    Embedding embed_tokens_{nullptr};
    torch::nn::ModuleList blocks_{nullptr};
    std::vector<LlamaDecoderLayer> layers_;
    LlamaRMSNorm norm_{nullptr};
public:
    LlamaModelImpl(const LlamaConfig& config, const torch::TensorOptions& options){
        embed_tokens_ = register_module("embed_tokens", Embedding(config.vocab_size, config.hidden_size, options));
        blocks_ = register_module("layers", torch::nn::ModuleList());
        for(int i = 0;i < config.n_layers;i ++){
            auto block = LlamaDecoderLayer(config, options, i);
            layers_.push_back(block);
            blocks_->push_back(block);
        }
        norm_ = register_module("norm", LlamaRMSNorm(config.hidden_size, config.rms_norm_eps, options));
    }
    torch::Tensor forward(torch::Tensor input_ids, torch::Tensor position_ids, const ModelParameters& model_params){
        auto hidden_states = embed_tokens_(input_ids);
        for(int i = 0;i < layers_.size();i ++){
            auto& layer = layers_[i];
            hidden_states = layer(hidden_states, position_ids, model_params);
        }
        return norm_(hidden_states);
    }
};
TORCH_MODULE(LlamaModel);

class LlamaForCausalLMImpl : public torch::nn::Module{
public:
    LlamaModel model_{nullptr};
    Linear lm_head_{nullptr};
public:
    LlamaForCausalLMImpl(const LlamaConfig& config, const torch::TensorOptions& options){
        model_ = register_module("model", LlamaModel(config, options));
        lm_head_ = register_module("lm_head", Linear(config.hidden_size, config.vocab_size, false, options));
    }
    torch::Tensor forward(torch::Tensor input_ids, torch::Tensor position_ids, const ModelParameters& model_params){
        auto hidden_state = model_(input_ids, position_ids, model_params);
        auto logits = lm_head_(hidden_state);
        return logits;
    }
};
TORCH_MODULE(LlamaForCausalLM);

}