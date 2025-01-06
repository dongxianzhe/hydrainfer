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

    {
    // 1. g2s
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

    {
    // 2. s2r
    Tensor ra = make_tensor<half>(make_shape(Int<8>{}, Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<8>{}, Int<16>{}));
    Tensor rb = make_tensor<half>(make_shape(Int<4>{}, Int<16>{}, Int<2>{}), make_stride(Int<1>{}, Int<4>{}, Int<64>{}));
    Tensor rc = make_tensor<half>(make_shape(Int<4>{}, Int<2>{}, Int<16>{}), make_stride(Int<1>{}, Int<4>{}, Int<8>{}));
    Tensor s2r_sa = make_tensor(
        make_smem_ptr(ashm), 
        make_shape (Int<2>{}, Int<2>{}     , Int<2>{}, Int<2>{}      , Int<2>{} , Int<4>{}, Int<8>{} , Int<4>{}     ), 
        make_stride(Int<1>{}, Int<8 * 32>{}, Int<8>{}, Int<64 * 32>{}, Int<16>{}, Int<2>{}, Int<32>{}, Int<16 * 32>{})
        );
    Tensor s2r_ra = make_tensor(
        make_rmem_ptr(ra.data()),
        make_shape(Int<2>{}, Int<2>{}, Int<2>{}, Int<2>{}, Int<2>{}), 
        make_stride(Int<1>{}, Int<2>{}, Int<4>{}, Int<8>{}, Int<16>{})
        );
    Tensor s2r_sb = make_tensor(
        make_smem_ptr(bshm), 
        make_shape (Int<2>{}, Int<2>{}, Int<16>{}    , Int<2>{} , Int<4>{}, Int<8>{} , Int<4>{}), 
        make_stride(Int<1>{}, Int<8>{}, Int<8 * 32>{}, Int<16>{}, Int<2>{}, Int<32>{}, Int<0>{})
    );
    Tensor s2r_rb = make_tensor(
        make_rmem_ptr(rb.data()), 
        make_shape (Int<2>{}, Int<2>{}, Int<16>{}, Int<2>{}), 
        make_stride(Int<1>{}, Int<2>{}, Int<4>{} , Int<64>{})
    );
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    for(int x = 0;x < 2;x ++){
        for(int y = 0;y < 2;y ++){
            for(int i = 0;i < 2;i ++){
                for(int j = 0;j < 2;j ++){
                    for(int k = 0;k < 2;k ++){
                        s2r_ra(k, j, i, x, y) = s2r_sa(k, j, i, x, y, lane_id % 4, lane_id / 4, warp_id);                   
                    }
                }
            }
        }
    }
    for(int x = 0;x < 16;x ++){
        for(int y = 0;y < 2;y ++){
            for(int i = 0;i < 2;i ++){
                for(int j = 0;j < 2;j ++){
                    s2r_rb(j, i, x, y) = s2r_sb(j, i, x, y, lane_id % 4, lane_id / 4, warp_id);
                }
            }
        }
    }
    clear(rc);
    __syncthreads();
    // 3. compute
    auto mma_atom = SM80_16x8x16_F16F16F16F16_TN{};
    auto thr_layout = make_layout(make_shape(Int<4>{}, Int<1>{}, Int<1>{}));  // m n k
    auto permutations = Tile<Int<64>, Int<32>, Int<16>>{}; // todo is there make_tile?
    auto tiled_mma = make_tiled_mma(mma_atom, thr_layout, permutations);
    for(int ik = 0;ik < 2; ik ++){
        cute::gemm(tiled_mma, rc, ra(_, _, ik), rb(_, _, ik), rc);
    }
    __syncthreads();
    // 4. r2s
    Tensor r2s_rc = make_tensor(
        make_rmem_ptr(rc.data()), 
        make_shape (Int<2>{}, Int<2>{}, Int<2>{}, Int<16>{}), 
        make_stride(Int<1>{}, Int<2>{}, Int<4>{}, Int<8>{} )
        );
    Tensor r2s_sc = make_tensor(
        make_smem_ptr(cshm), 
        make_shape (Int<2>{}, Int<2>{}      , Int<2>{}       , Int<16>{}, Int<4>{}, Int<8>{}  , Int<4>{}       ), 
        make_stride(Int<1>{}, Int<8 * 128>{}, Int<64 * 128>{}, Int<8>{} , Int<2>{}, Int<128>{}, Int<16 * 128>{})
    );
    for(int m = 0;m < 2;m ++){
        for(int n = 0;n < 16;n ++){
            for(int i = 0;i < 2;i ++){
                for(int j = 0;j < 2;j ++){
                    r2s_sc(j, i, m, n, lane_id % 4, lane_id / 4, warp_id) = r2s_rc(j, i, m, n);
                }
            }
        }
    }
    __syncthreads();
    }

    {
    // 5. s2g
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