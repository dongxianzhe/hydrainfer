#pragma once
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

namespace mllm{

template <class Engine1, class Engine2, class Layout1, class Layout2>
__device__ void tile_g2s_copy_async(cute::Tensor<Engine1, Layout1> ga, cute::Tensor<Engine2, Layout2> sa){
    // a (128, 32) : (x, 1)
    // b (128, 32) : (x, 1)
    // 128 threads
    using namespace cute;
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, half>;

    auto g2s_tiled_copy_a = make_tiled_copy(
        g2s_copy_atom{}, 
        make_layout(
            make_shape(Int<32>{}, Int<4>{}),
            make_stride(Int<4>{}, Int<1>{})
        ), // thread layout
        make_layout(make_shape(Int<1>{}, Int<8>{})) // data tile layout each thread copy
    );

    int idx = threadIdx.x;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto taga_copy = g2s_thr_copy_a.partition_S(ga); // (8, 128 / 32, 32 / 4)
    auto tasa_copy = g2s_thr_copy_a.partition_D(sa); // (8, 128 / 32, 32 / 4)

    cute::copy(g2s_tiled_copy_a, taga_copy(_, _, _), tasa_copy(_, _, _));
}




template<int n_tokens, int hidden_size>
__global__ void linear_smem_v2_kernel(half* hptr, half* wptr, half* optr){
    constexpr int BLOCK_SIZE_K = 32;
    constexpr int BLOCK_SIZE_M = 128;
    constexpr int BLOCK_SIZE_N = 128;
    using namespace cute;
    Tensor h = make_tensor(make_gmem_ptr(hptr), make_shape(Int<n_tokens>{}, Int<hidden_size>{}), make_stride(Int<hidden_size>{}, Int<1>{}));
    Tensor w = make_tensor(make_gmem_ptr(wptr), make_shape(Int<hidden_size>{}, Int<hidden_size>{}), make_stride(Int<hidden_size>{}, Int<1>{}));
    Tensor o = make_tensor(make_gmem_ptr(optr), make_shape(Int<n_tokens>{}, Int<hidden_size>{}), make_stride(Int<hidden_size>{}, Int<1>{}));
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
        tile_g2s_copy_async(gh(make_coord(_, _, bk)), sh);
        tile_g2s_copy_async(gw(make_coord(_, _, bk)), sw);
        cp_async_fence();
        cp_async_wait<0>();
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

torch::Tensor linear_smem_v2(torch::Tensor h, torch::Tensor w){
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
    linear_smem_v2_kernel<n_tokens, hidden_size><<<gridDim, blockDim>>>(
        static_cast<half*>(h.data_ptr()), 
        static_cast<half*>(w.data_ptr()),
        static_cast<half*>(o.data_ptr())
    );
    return o;
}

}