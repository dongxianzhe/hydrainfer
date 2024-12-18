#include <filesystem>
#include <algorithm>
#include <iostream>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <c10/core/TensorOptions.h>
#include "linear.h"
#include "llava.h"

namespace mllm{

TEST(model, llavaforcausal){
    std::string model_weights_path = "/home/xzd/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/8c85e9a4d626b7b908448be32c1ba5ad79b95e76";
    std::vector<std::string> model_weights_files;
    for (const auto& entry : std::filesystem::directory_iterator(model_weights_path)) {
        if (entry.path().extension() == ".safetensors") {
            model_weights_files.push_back(entry.path().string());
        }
    }

    LlavaConfig config;
    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    LlavaForConditionalGeneration model(config, options);

    auto dict = model->named_parameters();
    for(int i = 0;i < model_weights_files.size();i ++){
        auto state_dict = StateDict::load_safetensors(model_weights_files[i]);

        for(auto it = state_dict->begin(); it != state_dict->end(); it ++){
            std::string key = it->first;
            torch::Tensor weight = it->second;
            dict[key].copy_(weight);
        }
    }
    
    torch::TensorOptions int_options = torch::dtype(torch::kInt).device(torch::kCUDA);
    auto input_ids = torch::tensor({1, 22172}, int_options); // "hello"
    auto position_ids = torch::tensor({0, 1}, int_options);
    ModelParameters model_params;
    std::vector<std::unique_ptr<KVCache>> kv_caches;
    for(int i = 0;i < config.text_config.n_layers;i ++){
        kv_caches.emplace_back(std::make_unique<KVCache>(100, 16, config.text_config.n_kv_heads, config.text_config.head_size, options));
    }
    for(int i = 0;i < config.text_config.n_layers;i ++){
        AttentionParameters params;
        params.kv_cache = kv_caches[i].get();
        params.q_cu_seq_lens = torch::tensor({0, 2}, int_options);
        params.k_cu_seq_lens = torch::tensor({0, 2}, int_options);
        params.paged_kv_last_page_len = torch::tensor({2}, int_options);
        params.new_cache_slots = torch::tensor({0, 1}, int_options);
        params.block_tables = torch::tensor({0}, int_options);
        params.cu_block_lens = torch::tensor({0, 1}, int_options);
        params.num_sequences = 1;
        params.all_sequences_decode = false;
        params.q_max_seq_len = 128;
        params.k_max_seq_len = 128;
        model_params.attention_params.push_back(params);
    }
    torch::Tensor pixel_values;
    torch::Tensor image_features;
    auto logits = model(input_ids, pixel_values, image_features, position_ids, model_params);
    auto next_tokens_ids = torch::argmax(logits, -1, false);
    std::cout << next_tokens_ids << std::endl;
    EXPECT_EQ(next_tokens_ids[0].item<int>(), 23196);
    EXPECT_EQ(next_tokens_ids[1].item<int>(), 29991);
    // [23196, 29991]
}

}