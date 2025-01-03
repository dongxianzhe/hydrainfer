#include <gtest/gtest.h>
#include <torch/torch.h>

extern torch::Tensor inc(torch::Tensor a, int delta);

TEST(inc, inc_kernel){
    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    
    const int n_blocks = 128;
    const int warp_size = 32;
    const int delta = 10;
    auto a = torch::randn({n_blocks, warp_size}, options);
    auto o_ref = a + delta;
    auto o = inc(a, delta);

    EXPECT_TRUE(torch::allclose(o, o_ref, 1e-3, 1e-3));
}
