#include "tile_linear.h"
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

namespace mllm{

__global__ void tile_linear_kernel(half* hptr, half* wptr, half* optr){
    using namespace cute;
    Tensor gh = make_tensor(make_gmem_ptr(hptr), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor gw = make_tensor(make_gmem_ptr(wptr), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor go = make_tensor(make_gmem_ptr(optr), make_shape(Int<128>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}));
    int i = threadIdx.x;
    for(int j = 0;j < 128;j ++){
        half sum = 0;
        for(int k = 0;k < 32;k ++){
            sum += gh(i, k) * gw(j, k);
        }
        go(i, j) = sum;
    }
}

torch::Tensor tile_linear(torch::Tensor a, torch::Tensor b){
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 32;

    TORCH_CHECK(a.dtype() == torch::kHalf);
    TORCH_CHECK(b.dtype() == torch::kHalf);
    TORCH_CHECK(a.is_cuda());
    TORCH_CHECK(b.is_cuda());
    TORCH_CHECK(a.is_contiguous());
    TORCH_CHECK(b.is_contiguous());
    TORCH_CHECK(a.dim() == 2);
    TORCH_CHECK(b.dim() == 2);
    TORCH_CHECK(a.size(0) == M);
    TORCH_CHECK(a.size(1) == K);
    TORCH_CHECK(b.size(0) == N);
    TORCH_CHECK(b.size(1) == K);

    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    auto c = torch::zeros({M, N}, options);

    dim3 gridDim{1};
    dim3 blockDim{128};
    tile_linear_kernel<<<gridDim, blockDim>>>(
        static_cast<half*>(a.data_ptr()), 
        static_cast<half*>(b.data_ptr()),
        static_cast<half*>(c.data_ptr())
    );
    return c;
}

}