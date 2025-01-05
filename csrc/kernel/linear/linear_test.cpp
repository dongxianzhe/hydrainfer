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
    auto o1 = mllm::linear_naive(h, w);
    auto o = mllm::linear(h, w);

    std::cout << o_ref.view({-1}).slice(0, 0, 4) << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << o.view({-1}).slice(0, 0, 4) << std::endl;

    auto abs_error = torch::abs(o - o_ref).view({-1});
    int max_abs_error_id = torch::argmax(abs_error).item<int>();

    const float atol = 0.8;
    const float rtol = 0.1;
    std::cout << "max abs error: " << abs_error.max().item<float>() << 
        " at " << max_abs_error_id << " " << 
        o.view({-1})[max_abs_error_id].item<float>() << " " << o_ref.view({-1})[max_abs_error_id].item<float>() << std::endl;

    auto rel_error = (torch::abs(o - o_ref).view({-1}) - atol) / torch::abs(o_ref).view({-1});
    int max_rel_error_id = torch::argmax(rel_error).item<int>();
    std::cout << "max rel error: " << rel_error.max().item<float>() << 
        " at " << max_rel_error_id << " " << 
        o.view({-1})[max_rel_error_id].item<float>() << " " << o_ref.view({-1})[max_rel_error_id].item<float>() << std::endl;
    EXPECT_TRUE(torch::allclose(o1, o_ref, rtol, atol));
    EXPECT_TRUE(torch::allclose(o , o_ref, rtol, atol));
}

}