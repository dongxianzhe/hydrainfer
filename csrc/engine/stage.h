#pragma once
#include <vector>
#include <torch/torch.h>

namespace mllm{

struct Stage{

};

struct Fill : public Stage{
    std::vector<int> token_ids;
    std::vector<int> position_ids;
    std::vector<int> cache_ids;
    bool sample;
    int sample_dst_stage_id;
};

struct TextFill : public Fill{

};

struct ImageFill : public Fill{
    torch::Tensor pixel_values;
};

struct ImageEmbedFill : public Fill{
    torch::Tensor image_features;
};

struct ImageEmbed : public Stage{
    torch::Tensor pixel_values;
    int image_feature_dst_stage_id;
};

struct EmptyStage : public Stage{

};
}