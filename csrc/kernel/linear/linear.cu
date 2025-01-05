#include"linear.h"
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

namespace mllm{

template<int n_tokens, int hidden_size>
__global__ void linear_naive_kernel(half* h, half* w, half* o){
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

torch::Tensor linear_naive(torch::Tensor h, torch::Tensor w){
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
    linear_naive_kernel<n_tokens, hidden_size><<<gridDim, blockDim>>>(
        static_cast<half*>(h.data_ptr()), 
        static_cast<half*>(w.data_ptr()),
        static_cast<half*>(o.data_ptr())
    );
    return o;
}

template<int n_tokens, int hidden_size>
__global__ void linear_kernel(half* hptr, half* wptr, half* optr){
    constexpr int BLOCK_SIZE_K = 32;
    constexpr int BLOCK_SIZE_M = 128;
    constexpr int BLOCK_SIZE_N = 128;
    using namespace cute;
    Tensor h = make_tensor(make_gmem_ptr(hptr), make_shape(n_tokens, hidden_size), make_stride(hidden_size, 1));
    Tensor w = make_tensor(make_gmem_ptr(wptr), make_shape(hidden_size, hidden_size), make_stride(hidden_size, 1));
    Tensor o = make_tensor(make_gmem_ptr(optr), make_shape(n_tokens, hidden_size), make_stride(hidden_size, 1));
    Tensor gh = local_tile(h, make_tile(Int<BLOCK_SIZE_M>{}, Int<BLOCK_SIZE_K>{}), make_coord(blockIdx.x, _));
    Tensor gw = local_tile(w, make_tile(Int<BLOCK_SIZE_N>{}, Int<BLOCK_SIZE_K>{}), make_coord(blockIdx.y, _));
    Tensor go = local_tile(o, make_tile(Int<BLOCK_SIZE_M>{}, Int<BLOCK_SIZE_N>{}), make_coord(blockIdx.x, blockIdx.y));

    __shared__ half hshm[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ half wshm[BLOCK_SIZE_N * BLOCK_SIZE_K];
    __shared__ half oshm[BLOCK_SIZE_M * BLOCK_SIZE_N];
    Tensor sh = make_tensor(make_smem_ptr(hshm), make_shape(Int<BLOCK_SIZE_M>{}, Int<BLOCK_SIZE_K>{}), make_stride(Int<BLOCK_SIZE_K>{}, Int<1>{}));
    Tensor sw = make_tensor(make_smem_ptr(wshm), make_shape(Int<BLOCK_SIZE_N>{}, Int<BLOCK_SIZE_K>{}), make_stride(Int<BLOCK_SIZE_K>{}, Int<1>{}));
    Tensor so = make_tensor(make_smem_ptr(oshm), make_shape(Int<BLOCK_SIZE_M>{}, Int<BLOCK_SIZE_N>{}), make_stride(Int<BLOCK_SIZE_N>{}, Int<1>{}));

    constexpr int NUM_BLOCK_K = hidden_size / BLOCK_SIZE_K;
    for(int bk = 0; bk < NUM_BLOCK_K; bk ++){
        Tensor gh_tile = gh(make_coord(_, _, bk));
        Tensor gw_tile = gw(make_coord(_, _, bk));
        for(int k = 0;k < BLOCK_SIZE_K;k ++){
            sh(threadIdx.x, k) = gh_tile(threadIdx.x, k);
            sw(threadIdx.x, k) = gw_tile(threadIdx.x, k);
        }
        __syncthreads();

        int i = threadIdx.x;
        for(int j = 0;j < BLOCK_SIZE_N;j ++){
            half sum = 0;
            for(int k = 0;k < BLOCK_SIZE_K;k ++){
                sum += sh(i, k) * sw(j, k);
            }
            if(bk == 0){
                so(i, j) = sum;
            }else{
                so(i, j) = so(i, j) + sum;
            }
        }
        __syncthreads();
    }
    int i = threadIdx.x;
    for(int j = 0;j < BLOCK_SIZE_N;j ++){
        go(i, j) = so(i, j);
    }
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

    constexpr int BLOCK_SIZE_M = 128;
    constexpr int BLOCK_SIZE_N = 128;
    dim3 gridDim{n_tokens / BLOCK_SIZE_M, hidden_size / BLOCK_SIZE_N};
    dim3 blockDim{128};
    linear_kernel<n_tokens, hidden_size><<<gridDim, blockDim>>>(
        static_cast<half*>(h.data_ptr()), 
        static_cast<half*>(w.data_ptr()),
        static_cast<half*>(o.data_ptr())
    );
    return o;
}

}