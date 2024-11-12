#include <ATen/cuda/CUDAContext.h>
#include<iostream>
#include<torch/torch.h>
#include"reduce/reduce_sum.cuh"
#include"dispatch.h"

// calculate the root mean square norm.
// equation: x -> w * x / sqrt(E[x^2] + eps)
// The mean is calculated over the last dimension
// equilvalent to layernorm module in the T5 style No bias and no subtraction of
// mean.
template <typename T>
__global__ void rms_norm_kernel(T* __restrict__ out,
                                const T* __restrict__ input,
                                const T* __restrict__ weight,
                                const float epsilon,
                                int64_t n) {
  const auto tidx = threadIdx.x;
  const auto bidx = blockIdx.x;

  __shared__ float s_variance;
  float variance = 0.0f;

  for (int64_t i = tidx; i < n; i += blockDim.x) {
    const float x = input[bidx * n + i];
    variance += x * x;
  }
  variance = block_reduce_sum<float>(variance);
  if (tidx == 0) {
    s_variance = rsqrtf(variance / n + epsilon);
  }
  __syncthreads();

  for (int64_t i = tidx; i < n; i += blockDim.x) {
    const int64_t idx = bidx * n + i;
    const float x = input[idx];
    out[idx] = (T)(x * s_variance) * weight[i];
  }
}

void rms_norm(torch::Tensor& out,
              torch::Tensor input,
              torch::Tensor weight,
              float epsilon) {
  DCHECK(input.is_contiguous()) << "input tensor must be contiguous";
  DCHECK(out.is_contiguous()) << "output tensor must be contiguous";

  const int64_t n = input.size(1);

  dim3 grid(input.size(0));
  dim3 block(std::min<int>(n, 1024));
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(input.scalar_type(), scalar_t, [&] {
    rms_norm_kernel<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<scalar_t*>(out.data_ptr()),
            static_cast<scalar_t*>(input.data_ptr()),
            static_cast<scalar_t*>(weight.data_ptr()),
            epsilon,
            n);
  });
}