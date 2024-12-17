#include "stage.h"
#include <gtest/gtest.h>

namespace  mllm
{

TEST(stage, image_embed_fill){
    auto image_features = torch::randn({576, 4096});
    ImageEmbedFill stage;
    stage.image_features = image_features;

    std::vector<int> token_ids;
    token_ids.reserve(10000000);
    for(int i = 0;i < 10000000;i ++)token_ids.push_back(i);
    stage.token_ids = std::move(token_ids);
}

}