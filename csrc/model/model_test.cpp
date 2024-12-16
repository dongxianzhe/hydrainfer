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
    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    LlavaConfig config;
    LlavaForConditionalGeneration model(config, options);
}

