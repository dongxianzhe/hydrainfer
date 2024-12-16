#include "fake_model.h"
#include <gtest/gtest.h>

namespace mllm{

TEST(fakemodel, forward){
    int vocab_size = 32064;
    int n_tokens = 10;

    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    FakeModel model(32064, options);

    auto input_ids = torch::arange(n_tokens, options.dtype(torch::kInt32));
    auto position_ids = torch::arange(n_tokens, options.dtype(torch::kInt32));
    auto logits = model(input_ids, position_ids);
    std::cout << logits.sizes() << std::endl;
}

}