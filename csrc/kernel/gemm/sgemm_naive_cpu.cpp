#include<torch/torch.h>
#include <torch/types.h>

// A (M, K) : (K, 1)
// B (N, K) : (K, 1)
// C (M, N) : (N, 1)
void sgemm_naive_cpu_kernel(float* A, float* B, float* C, int M, int N, int K){
    for(int y = 0;y < M;y ++){
        for(int x = 0;x < N;x ++){
            float sum = 0;
            for(int k  = 0;k < K;k ++){
                sum += A[y * K + k] * B[x * K + k];
            }
            C[y * M + x] = sum;
        }
    }
}

void sgemm_naive_cpu(torch::Tensor& C, torch::Tensor A, torch::Tensor B){
    TORCH_CHECK(A.is_cpu(), "A must be on cpu");
    TORCH_CHECK(B.is_cpu(), "A must be on cpu");

    TORCH_CHECK(A.dtype() == torch::kFloat32, "sgemm_naive_cpu A only support float type");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "sgemm_naive_cpu B only support float type");

    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(1), "shape is invalid");

    // todo check shape and stride
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    sgemm_naive_cpu_kernel(
        static_cast<float*>(A.data_ptr()), 
        static_cast<float*>(B.data_ptr()), 
        static_cast<float*>(C.data_ptr()), 
        M, N, K
    );
}