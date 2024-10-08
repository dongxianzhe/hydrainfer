#include<torch/torch.h>
#include <torch/types.h>

#define div_ceil(x, mod) (((x) + (mod) - 1) / (mod))

// A (M, K) : (K, 1)
// B (N, K) : (K, 1)
// C (M, N) : (N, 1)
template<int BLOCK_SIZE_N, int BLOCK_SIZE_M>
__global__ void sgemm_naive_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int M, int N, int K){
    int x = blockIdx.x * BLOCK_SIZE_N + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE_M + threadIdx.y;
    if(x >= N || y >= M)return;
    float sum = 0;
    for(int k = 0;k < K;k ++){
        sum += A[y * K + k] * B[x * K + k];
    }
    C[y * N + x] = sum;
}

void sgemm_naive(torch::Tensor& C, torch::Tensor A, torch::Tensor B){
    TORCH_CHECK(A.is_cuda(), "A must be on cuda");
    TORCH_CHECK(B.is_cuda(), "B must be on cuda");

    TORCH_CHECK(A.dtype() == torch::kFloat32, "A only support float type");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B only support float type");

    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(1), "shape is invalid");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    const int BLOCK_SIZE_M = 8;
    const int BLOCK_SIZE_N = 8;
    const int NUM_BLOCK_M = div_ceil(M, BLOCK_SIZE_M);
    const int NUM_BLOCK_N = div_ceil(N, BLOCK_SIZE_N);
    dim3 gridDim(NUM_BLOCK_N, NUM_BLOCK_M);
    dim3 blockDim(BLOCK_SIZE_N, BLOCK_SIZE_M);
    sgemm_naive_kernel<BLOCK_SIZE_N, BLOCK_SIZE_M><<<gridDim, blockDim>>>(
        static_cast<float*>(A.data_ptr()), 
        static_cast<float*>(B.data_ptr()), 
        static_cast<float*>(C.data_ptr()),
        M, N, K
    );
}

#define DISPATCH_CASE(name, value, ...) \
    case value: { \
        constexpr int name = value; \
        __VA_ARGS__(); \
        break; \
    }

#define DISPATCH_BLOCK_SIZE(constname, name, ...) \
    switch (name){ \
        DISPATCH_CASE(constname, 1, __VA_ARGS__) \
        DISPATCH_CASE(constname, 2, __VA_ARGS__) \
        DISPATCH_CASE(constname, 4, __VA_ARGS__) \
        DISPATCH_CASE(constname, 8, __VA_ARGS__) \
        DISPATCH_CASE(constname, 16, __VA_ARGS__) \
        DISPATCH_CASE(constname, 32, __VA_ARGS__) \
        default:{ \
           fprintf(stderr, "invalid dispatch %s",  #constname); \
           exit(1); \
        } \
    }


void sgemm_naive_dispatchblocksize(torch::Tensor& C, torch::Tensor A, torch::Tensor B, int block_size_n, int block_size_m){
    TORCH_CHECK(A.is_cuda(), "A must be on cuda");
    TORCH_CHECK(B.is_cuda(), "B must be on cuda");

    TORCH_CHECK(A.dtype() == torch::kFloat32, "A only support float type");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B only support float type");

    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(1), "shape is invalid");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    DISPATCH_BLOCK_SIZE(BLOCK_SIZE_N, block_size_n, [&]{
        DISPATCH_BLOCK_SIZE(BLOCK_SIZE_M, block_size_m, [&]{
            const int NUM_BLOCK_M = div_ceil(M, BLOCK_SIZE_M);
            const int NUM_BLOCK_N = div_ceil(N, BLOCK_SIZE_N);
            dim3 gridDim(NUM_BLOCK_N, NUM_BLOCK_M);
            dim3 blockDim(BLOCK_SIZE_N, BLOCK_SIZE_M);
            sgemm_naive_kernel<BLOCK_SIZE_N, BLOCK_SIZE_M><<<gridDim, blockDim>>>(
                static_cast<float*>(A.data_ptr()), 
                static_cast<float*>(B.data_ptr()), 
                static_cast<float*>(C.data_ptr()),
                M, N, K
            );
        });
    });
}