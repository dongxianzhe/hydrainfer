#pragma once
#include "clip.h"
#include "llama.h"
#include "model_loader/state_dict.h"

namespace mllm{

class CLIPEncoderImpl : public torch::nn::Module{
public:
    torch::nn::ModuleList blocks_{nullptr};
    std::vector<CLIPEncoderLayer> layers_;
    int n_layers_;
public:
    CLIPEncoderImpl(const CLIPConfig& config, const torch::TensorOptions& options) : n_layers_(config.n_layers){
        blocks_ = register_module("layers", torch::nn::ModuleList());
        for(int i = 0;i < config.n_layers;i ++){
            auto block = CLIPEncoderLayer(config, options, i);
            layers_.push_back(block);
            blocks_->push_back(block);
        }
    }
    torch::Tensor forward(torch::Tensor h, int vision_feature_layer){
        int n_blocks = (vision_feature_layer + n_layers_) % n_layers_ + 1;
        for(int i = 0;i < n_blocks;i ++){
            auto& layer = layers_[i];
            h = layer(h);
        }
        return h;
    }
};
TORCH_MODULE(CLIPEncoder);


class CLIPVisionTransformerImpl : public torch::nn::Module{
public:
    CLIPVisionEmbeddings embeddings_{nullptr};
    LayerNorm pre_layrnorm_{nullptr};
    CLIPEncoder encoder_{nullptr};
    LayerNorm post_layernorm_{nullptr};
public:
    CLIPVisionTransformerImpl(const CLIPConfig& config, const torch::TensorOptions& options){
        embeddings_ = register_module("embeddings", CLIPVisionEmbeddings(config, options));
        pre_layrnorm_ = register_module("pre_layrnorm", LayerNorm(config.hidden_size, config.layer_norm_eps, true, options));
        encoder_ = register_module("encoder", CLIPEncoder(config, options));
        post_layernorm_ = register_module("post_layernorm", LayerNorm(config.hidden_size, config.layer_norm_eps, true, options));
    }
    torch::Tensor forward(torch::Tensor pixel_values, int vision_feature_layer){
        auto h = embeddings_(pixel_values);
        h = pre_layrnorm_(h);
        h = encoder_(h, vision_feature_layer);
        return h;
    }
};
TORCH_MODULE(CLIPVisionTransformer);

class CLIPVisionModelImpl : public torch::nn::Module{
    CLIPVisionTransformer vision_model_{nullptr};
public:
    CLIPVisionModelImpl(const CLIPConfig& config, const torch::TensorOptions& options){
        vision_model_ = register_module("vision_model", CLIPVisionTransformer(config, options));
    }
    torch::Tensor forward(torch::Tensor pixel_values, int vision_feature_layer){
        return vision_model_(pixel_values, vision_feature_layer);
    }
};
TORCH_MODULE(CLIPVisionModel);

class LlavaConfig{
public:
    CLIPConfig vision_config;
    LlamaConfig text_config;
    int vision_feature_layer = -2;

    // language model config
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
    // vision model config
    int n_image_tokens = 576;
    int image_token_id = 32000;
};

class LlavaMultiModalProjectorImpl : public torch::nn::Module{
public:
    Linear linear_1_{nullptr};
    Linear linear_2_{nullptr};
    torch::nn::GELU act_{nullptr};
public:
    LlavaMultiModalProjectorImpl(const LlavaConfig& config, const torch::TensorOptions& options){
        linear_1_ = register_module("linear_1", Linear(config.vision_config.hidden_size, config.text_config.hidden_size, true, options));
        linear_2_ = register_module("linear_2", Linear(config.text_config.hidden_size, config.text_config.hidden_size, true, options));
        act_ = register_module("act", torch::nn::GELU());
    }
    torch::Tensor forward(torch::Tensor image_features){
        auto hidden_states = linear_1_(image_features);
        hidden_states = act_(hidden_states);
        hidden_states = linear_2_(hidden_states);
        return hidden_states;
    }
};
TORCH_MODULE(LlavaMultiModalProjector);


class LlavaForConditionalGenerationImpl : public torch::nn::Module{
public:
    CLIPVisionModel vision_tower_{nullptr};
    LlavaMultiModalProjector multi_modal_projector_{nullptr};
    LlamaForCausalLM language_model_{nullptr};
    int vision_feature_layer_;
    int image_token_index_;
public:
    LlavaForConditionalGenerationImpl(const LlavaConfig& config, const torch::TensorOptions& options) : vision_feature_layer_(config.vision_feature_layer), image_token_index_(config.vision_config.image_token_index){
        vision_tower_ = register_module("vision_tower", CLIPVisionModel(config.vision_config, options));
        multi_modal_projector_ = register_module("multi_modal_projector", LlavaMultiModalProjector(config, options));
        language_model_ = register_module("language_model", LlamaForCausalLM(config.text_config, options));
    }

    torch::Tensor input_embed(torch::Tensor input_ids){
        return language_model_->model_->embed_tokens_(input_ids);
    }

    torch::Tensor image_embed(torch::Tensor pixel_values, const ModelParameters& model_params){
        auto hidden_states = vision_tower_(pixel_values, vision_feature_layer_);
        auto selected_image_feature = hidden_states.index({torch::indexing::Slice(), torch::indexing::Slice(1)});
        auto image_features = multi_modal_projector_(selected_image_feature);
        return image_features;
    }

    torch::Tensor merge_embed(torch::Tensor input_embeds, torch::Tensor image_features, torch::Tensor image_overwrite_mask){
        if(torch::sum(image_overwrite_mask, false).item<int>() != image_features.view({-1, input_embeds.size(-1)}).size(0))return input_embeds;
        input_embeds.index_put_({image_overwrite_mask, torch::indexing::Slice()}, image_features.view({-1, input_embeds.size(-1)}));
        return input_embeds;
    }

    torch::Tensor forward(torch::Tensor input_ids, torch::Tensor pixel_values, torch::Tensor image_features, torch::Tensor position_ids, const ModelParameters& model_params){
        auto input_embeds = input_embed(input_ids);
        if(pixel_values.defined()){
            auto image_overwrite_mask = input_ids == image_token_index_;
            auto image_features = image_embed(pixel_values, model_params);
            input_embeds = merge_embed(input_embeds, image_features, image_overwrite_mask);
        }
        else if(image_features.defined()){
            auto image_overwrite_mask = input_ids == image_token_index_;
            input_embeds = merge_embed(input_embeds, image_features, image_overwrite_mask);
        }
        auto logits = language_model_(input_embeds, position_ids, model_params);

        return logits;
    }
};
TORCH_MODULE(LlavaForConditionalGeneration);


}