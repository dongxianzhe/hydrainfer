#include "linear.h"
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace mllm{

TEST(linear, linear_kernel){
    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    const int n_tokens = 4096;
    const int hidden_size = 4096;
    auto h = torch::randn({n_tokens, hidden_size}, options);
    auto w = torch::randn({hidden_size, hidden_size}, options);

    auto o_ref = torch::nn::functional::linear(h, w);
    auto o = mllm::linear(h, w);

    std::cout << o_ref.view({-1}).slice(0, 0, 4) << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << o.view({-1}).slice(0, 0, 4) << std::endl;

    EXPECT_TRUE(torch::allclose(o, o_ref, 1e-2, 1e-2));
}

}