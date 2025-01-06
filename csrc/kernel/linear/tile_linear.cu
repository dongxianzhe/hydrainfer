#include "tile_linear.h"
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

namespace mllm{

__global__ void tile_linear_kernel(half* aptr, half* bptr, half* cptr){
    using namespace cute;
    Tensor ga = make_tensor(make_gmem_ptr(aptr), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor gb = make_tensor(make_gmem_ptr(bptr), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor gc = make_tensor(make_gmem_ptr(cptr), make_shape(Int<128>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}));
    __shared__ half ashm [128 * 32];
    __shared__ half bshm [128 * 32];
    __shared__ half cshm [128 * 128];
    Tensor sa = make_tensor(make_gmem_ptr(ashm), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor sb = make_tensor(make_gmem_ptr(bshm), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor sc = make_tensor(make_gmem_ptr(cshm), make_shape(Int<128>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}));
    // 1. g2s
    {
        int i = threadIdx.x;
        for(int j = 0;j < 32;j ++)sa(i, j) = ga(i, j);
        for(int j = 0;j < 32;j ++)sb(i, j) = gb(i, j);
        for(int j = 0;j < 128;j ++)sc(i, j) = gc(i, j);
        __syncthreads();
    }

    // 2. compute
    {
        int i = threadIdx.x;
        for(int j = 0;j < 128;j ++){
            half sum = 0;
            for(int k = 0;k < 32;k ++){
                sum += ga(i, k) * gb(j, k);
            }
            sc(i, j) += sum;
        }
        __syncthreads();
    }
    // 3. s2g
    {
        int i = threadIdx.x;
        for(int j = 0;j < 128;j ++)gc(i, j) = sc(i, j);
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