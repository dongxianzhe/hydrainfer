#include <gtest/gtest.h>
#include <torch/torch.h>

TEST(linear, linear_kernel){
    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    const int n_tokens = 4096;
    const int hidden_size = 4096;
    auto h = torch::randn({n_tokens, hidden_size}, options);
    auto w = torch::randn({hidden_size, hidden_size}, options);

    auto o_ref = torch::nn::functional::linear(h, w);
    auto o = torch::matmul(h, w.transpose(0, 1));
    EXPECT_TRUE(torch::allclose(o, o_ref, 1e-3, 1e-3));
}