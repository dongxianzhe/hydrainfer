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
    Tensor sa = make_tensor(make_smem_ptr(ashm), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor sb = make_tensor(make_smem_ptr(bshm), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor sc = make_tensor(make_smem_ptr(cshm), make_shape(Int<128>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}));
    // 1. g2s
    {
        using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
        using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
        using g2s_copy_atom = Copy_Atom<g2s_copy_traits, half>;
        auto thr_layout = make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
        auto val_layout = make_layout(make_shape(Int<1>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
        auto g2s_tiled_copy = make_tiled_copy(g2s_copy_atom{}, thr_layout, val_layout);
        auto g2s_thr_copy = g2s_tiled_copy.get_slice(threadIdx.x);
        auto g2s_ga = g2s_thr_copy.partition_S(ga); // (8, 128 / 32 / 1, 32 / 4 / 8)
        auto g2s_sa = g2s_thr_copy.partition_D(sa); // (8, 128 / 32 / 1, 32 / 4 / 8)
        auto g2s_gb = g2s_thr_copy.partition_S(gb); // (8, 128 / 32 / 1, 32 / 4 / 8)
        auto g2s_sb = g2s_thr_copy.partition_D(sb); // (8, 128 / 32 / 1, 32 / 4 / 8)
        cute::copy(g2s_tiled_copy, g2s_ga(_, _, _), g2s_sa(_, _, _));
        cute::copy(g2s_tiled_copy, g2s_gb(_, _, _), g2s_sb(_, _, _));
        cp_async_fence();
        cp_async_wait<0>();
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
            sc(i, j) = sum;
        }
        __syncthreads();
    }
    // 3. s2g
    {
        auto s2g_copy_atom = Copy_Atom<UniversalCopy<cute::uint128_t>, half>{};
        auto thr_layout = make_layout(make_shape(Int<4>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
        auto val_layout = make_layout(make_shape(Int<1>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
        auto s2g_tiled_copy = make_tiled_copy(s2g_copy_atom, thr_layout, val_layout);
        auto s2g_thr_copy = s2g_tiled_copy.get_thread_slice(threadIdx.x);
        auto s2g_sc = s2g_thr_copy.partition_S(sc); // (8, 128 / 4 / 1, 128 / 32 / 8)
        auto s2g_gc = s2g_thr_copy.partition_D(gc); // (8, 128 / 4 / 1, 128 / 32 / 8)
        cute::copy(s2g_tiled_copy, s2g_sc(_, _, _), s2g_gc(_, _, _));
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