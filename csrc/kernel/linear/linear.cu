#include"linear.h"
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

namespace mllm{

template<int n_tokens, int hidden_size>
__global__ void linear_kernel(half* h, half* w, half* o){
    using namespace cute;
    Tensor gh = make_tensor(make_gmem_ptr(h), make_shape(n_tokens, hidden_size), make_stride(hidden_size, 1));
    Tensor gw = make_tensor(make_gmem_ptr(w), make_shape(hidden_size, hidden_size), make_stride(hidden_size, 1));
    Tensor go = make_tensor(make_gmem_ptr(o), make_shape(n_tokens, hidden_size), make_stride(hidden_size, 1));

    int i = blockIdx.x;
    int j = blockIdx.y;
    float sum = 0;
    for(int k = 0;k < hidden_size;k ++){
        sum += static_cast<float>(gh(i, k) * gw(j, k));
    }
    go(i, j) = sum;
}

torch::Tensor linear(torch::Tensor h, torch::Tensor w){
    constexpr int n_tokens = 4096;
    constexpr int hidden_size = 4096;

    TORCH_CHECK(h.dtype() == torch::kHalf);
    TORCH_CHECK(w.dtype() == torch::kHalf);
    TORCH_CHECK(h.is_cuda());
    TORCH_CHECK(w.is_cuda());
    TORCH_CHECK(h.is_contiguous());
    TORCH_CHECK(w.is_contiguous());
    TORCH_CHECK(h.dim() == 2);
    TORCH_CHECK(w.dim() == 2);
    TORCH_CHECK(h.size(0) == n_tokens);
    TORCH_CHECK(h.size(1) == hidden_size);
    TORCH_CHECK(w.size(0) == hidden_size);
    TORCH_CHECK(w.size(1) == hidden_size);

    auto o = torch::zeros_like(h);

    dim3 gridDim{n_tokens, hidden_size};
    dim3 blockDim{1};
    linear_kernel<n_tokens, hidden_size><<<gridDim, blockDim>>>(
        static_cast<half*>(h.data_ptr()), 
        static_cast<half*>(w.data_ptr()),
        static_cast<half*>(o.data_ptr())
    );
    return o;
}

}