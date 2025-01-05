#include<iostream>
#include<gtest/gtest.h>
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

TEST(cute, tensor){
    torch::Tensor t = torch::randn({2, 3});
    std::cout << t << std::endl;

    cute::Tensor tensor = cute::make_tensor(static_cast<float*>(t.data_ptr()), cute::make_shape(cute::Int<2>{}, cute::Int<3>{}), cute::make_stride(cute::Int<3>{}, cute::Int<1>{}));
    cute::print(tensor);std::cout << std::endl;
    
    for (int i = 0; i < cute::size<0>(tensor); ++i) {
        for (int j = 0; j < cute::size<1>(tensor); ++j) {
            tensor(i, j) ++;
        }
    }

    std::cout << t << std::endl;
}

TEST(cute, local_tile){
    using namespace cute;
    torch::Tensor a = torch::randn({1024, 256});
    Tensor t = make_tensor(
        static_cast<float*>(a.data_ptr()), 
        make_shape(Int<1024>{}, Int<256>{}), 
        make_stride(Int<256>{}, Int<1>{})
    );
    print(t);puts("");
    Tensor s1 = local_tile(t, make_tile(Int<128>{}, Int<32>{}), make_coord(_, _));
    Tensor s2 = local_tile(t, make_tile(Int<128>{}, Int<32>{}), make_coord(2, _));
    Tensor s3 = local_tile(t, make_tile(Int<128>{}, Int<32>{}), make_coord(_, 3));
    Tensor s4 = local_tile(t, make_tile(Int<128>{}, Int<32>{}), make_coord(2, 3));
    print(s1);puts("");
    print(s2);puts("");
    print(s3);puts("");
    print(s4);puts("");
}

TEST(cute, slice){
    using namespace cute;
    torch::Tensor a = torch::randn({1024, 256});
    Tensor t = make_tensor(
        static_cast<float*>(a.data_ptr()), 
        make_shape(Int<1024>{}, Int<256>{}), 
        make_stride(Int<256>{}, Int<1>{})
    );
    print(t);puts("");
    Tensor s = local_tile(t, make_tile(Int<128>{}, Int<32>{}), make_coord(2, _));
    print(s);puts("");
    auto s_slice = s(make_coord(_, _, 1));
    print(s_slice);puts("");
}

__global__ void shared_memory_kernel(){
    constexpr int len = 48 * 1024 / 4;
    __shared__ volatile float data[len];
    data[len - 1] = 1;
}

TEST(kernel, max_shared_memory){
    shared_memory_kernel<<<1024, 32>>>();
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
}

template <class Engine1, class Engine2, class Layout>
__device__ void tile_g2s_copy(cute::Tensor<Engine1, Layout> ga, cute::Tensor<Engine2, Layout> sa){
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
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
}

__global__ void g2s_copy_kernel(half* a, half* o){
    using namespace cute;    
    Tensor ga = make_tensor(make_gmem_ptr(a), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor go = make_tensor(make_gmem_ptr(o), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    __shared__ half ashm[128 * 32];
    Tensor sa = make_tensor(make_smem_ptr(ashm), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));

    tile_g2s_copy(ga, sa);

    // s2g
    int i = threadIdx.x;
    for(int j = 0;j < 32;j ++){
        go(i, j) = sa(i, j);
    }
}

TEST(cute, g2s_copy){
    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    torch::Tensor a = torch::randn({128, 32}, options);
    torch::Tensor o_ref = a.clone();
    torch::Tensor o = torch::zeros({128, 32}, options);
    g2s_copy_kernel<<<1, 128>>>(
        static_cast<half*>(a.data_ptr()), 
        static_cast<half*>(o.data_ptr())
    );
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    std::cout << o_ref.view({-1}).slice(0, 0, 4) << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << o.view({-1}).slice(0, 0, 4) << std::endl;
    EXPECT_TRUE(torch::allclose(o, o_ref));
}



__global__ void register_copy_kernel(half* aptr, half* optr){
    using namespace cute;
    Tensor r = make_tensor<half>(make_shape(Int<32>{}), make_stride(Int<1>{}));
    Tensor a = make_tensor(make_gmem_ptr(aptr), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor o = make_tensor(make_gmem_ptr(optr), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    int i = threadIdx.x;
    for(int j = 0;j < 32;j ++){
        r(j) =  a(i, j);
    }
    for(int j = 0;j < 32;j ++){
        o(i, j) = r(j);
    }
}

TEST(cute, register_tensor){
    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    auto a = torch::randn({128, 32}, options);
    auto o_ref = a.clone();
    auto o = torch::empty({128, 32}, options);
    register_copy_kernel<<<1, 128>>>(
        static_cast<half*>(a.data_ptr()), 
        static_cast<half*>(o.data_ptr())
    );
    EXPECT_TRUE(torch::allclose(o , o_ref));
}

__global__ void tiled_mma_kernel(half* a, half* b, half*c){
    using namespace cute;
    Tensor ga = make_tensor(make_gmem_ptr(a), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor gb = make_tensor(make_gmem_ptr(b), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor gc = make_tensor(make_gmem_ptr(c), make_shape(Int<128>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}));
    __shared__ half ashm[128 * 32];
    __shared__ half bshm[128 * 32];
    __shared__ half cshm[128 * 128];
    Tensor sa = make_tensor(make_smem_ptr(ashm), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor sb = make_tensor(make_smem_ptr(bshm), make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Tensor sc = make_tensor(make_smem_ptr(cshm), make_shape(Int<128>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}));
    int i = threadIdx.x;
    for(int j = 0;j < 32;j ++)sa(i, j) = ga(i, j);
    for(int j = 0;j < 32;j ++)sb(i, j) = gb(i, j);
    for(int j = 0;j < 128;j ++)sc(i, j) = 0;
    __syncthreads();
    // s2r
    Tensor ra = make_tensor<half>(make_shape(Int<8>{}, Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<8>{}, Int<16>{}));
    Tensor rb = make_tensor<half>(make_shape(Int<4>{}, Int<16>{}, Int<2>{}), make_stride(Int<1>{}, Int<4>{}, Int<64>{}));
    Tensor rc = make_tensor<half>(make_shape(Int<4>{}, Int<2>{}, Int<16>{}), make_stride(Int<1>{}, Int<4>{}, Int<8>{}));
    int sa_m_offset = threadIdx.x / 32 * 16 + threadIdx.x % 32 / 4;
    int sa_k_offset = threadIdx.x % 4 * 2;
    for(int i = 0;i < 2;i ++){
        for(int j = 0;j < 2;j ++){
            ra(0, i, j) = sa(i * 64 + sa_m_offset, j * 16 + sa_k_offset);
            ra(1, i, j) = sa(i * 64 + sa_m_offset, j * 16 + sa_k_offset + 1);
            ra(2, i, j) = sa(i * 64 + sa_m_offset + 8, j * 16 + sa_k_offset);
            ra(3, i, j) = sa(i * 64 + sa_m_offset + 8, j * 16 + sa_k_offset + 1);
            ra(4, i, j) = sa(i * 64 + sa_m_offset, j * 16 + sa_k_offset + 8);
            ra(5, i, j) = sa(i * 64 + sa_m_offset, j * 16 + sa_k_offset + 8 + 1);
            ra(6, i, j) = sa(i * 64 + sa_m_offset + 8, j * 16 + sa_k_offset + 8);
            ra(7, i, j) = sa(i * 64 + sa_m_offset + 8, j * 16 + sa_k_offset + 8 + 1);
        }
    }
    int sb_n_offset = threadIdx.x % 32 / 4;
    int sb_k_offset = threadIdx.x % 4 * 2;
    for(int i = 0;i < 16;i ++){
        for(int j = 0;j < 2;j ++){
            rb(0, i, j) = sb(i * 8 + sb_n_offset, j * 16 + sb_k_offset);
            rb(1, i, j) = sb(i * 8 + sb_n_offset, j * 16 + sb_k_offset + 1);
            rb(2, i, j) = sb(i * 8 + sb_n_offset, j * 16 + sb_k_offset + 8);
            rb(3, i, j) = sb(i * 8 + sb_n_offset, j * 16 + sb_k_offset + 9);
        }
    }
    clear(rc);
    __syncthreads();

    auto tiled_mma = make_tiled_mma(
        SM80_16x8x16_F16F16F16F16_TN{},
        make_layout(make_shape(Int<4>{}, Int<1>{}, Int<1>{})),  // m n k
        Tile<Int<64>, Int<32>, Int<16>>{} // todo is there make_tile?
    );
    // auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    // auto ra_ref = thr_mma.partition_fragment_A(ga);
    // auto rb_ref = thr_mma.partition_fragment_B(gb);
    // auto rc_ref = thr_mma.partition_fragment_C(gc);

    // if(thread(0, 0)){
    //     print(ra_ref);print("\n");
    //     print(rb_ref);print("\n");
    //     print(rc_ref);print("\n");
    // }

    for(int ik = 0;ik < 2; ik ++){
        cute::gemm(tiled_mma, rc, ra(_, _, ik), rb(_, _, ik), rc);
    }
    __syncthreads();

    int sc_m_offset = threadIdx.x / 32 * 16 + threadIdx.x % 32 / 4;
    int sc_n_offset = threadIdx.x % 4 * 2;
    for(int i = 0;i < 2;i ++){
        for(int j = 0;j < 16;j ++){
            sc(i * 64 + sc_m_offset, j * 8 + sc_n_offset) = rc(0, i, j);
            sc(i * 64 + sc_m_offset, j * 8 + sc_n_offset + 1) = rc(1, i, j);
            sc(i * 64 + sc_m_offset + 8, j * 8 + sc_n_offset) = rc(2, i, j);
            sc(i * 64 + sc_m_offset + 8, j * 8 + sc_n_offset + 1) = rc(3, i, j);
        }
    }
    __syncthreads();

    for(int j = 0;j < 128;j ++){
        gc(i, j) = sc(i, j);
    }
}

TEST(cute, tiled_mma_kernel){
    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    auto a = torch::randn({128, 32}, options);
    auto b = torch::randn({128, 32}, options);
    auto c_ref = torch::nn::functional::linear(a, b);
    auto c = torch::zeros({128, 128}, options);

    tiled_mma_kernel<<<1, 128>>>(
        static_cast<half*>(a.data_ptr()), 
        static_cast<half*>(b.data_ptr()), 
        static_cast<half*>(c.data_ptr())
    );
    cudaDeviceSynchronize();

    const float atol = 1;
    const float rtol = 0.1;
    std::cout << c_ref.view({-1}).slice(0, 0, 4) << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << c.view({-1}).slice(0, 0, 4) << std::endl;
    EXPECT_TRUE(torch::allclose(c , c_ref, rtol, atol));
}