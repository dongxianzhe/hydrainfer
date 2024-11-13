#include <ATen/cuda/CUDAContext.h>
#include <c10/core/TensorImpl.h>
#include <torch/torch.h>
#include <cuda_fp16.h>

#include "activation.h"
#include "dispatch.h"

/* Silu Activation */

template <typename T>
struct SiluActivation {
  using return_type = T;
  static __device__ __forceinline__ T apply(const T& val) {
    // x * sigmoid(x)
    return (T)((float)val / (1.0f + __expf((float)-val)));
  }
};

template <template <typename T> class Activation, typename T>
__global__ void activation_kernel(T* __restrict__ out,
                                  const T* __restrict__ input,
                                  int n,
                                  int stride) {
  const uint32_t src_base_idx = blockIdx.x * stride;
  const uint32_t dst_base_idx = blockIdx.x * n;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const T x = __ldg(&input[src_base_idx + i]);
    out[dst_base_idx + i] = Activation<T>::apply(x);
  }
}

template <template <typename T> class Activation>
void launch_activation(torch::Tensor& out, torch::Tensor input) {
  const int n = static_cast<int>(input.size(1));
  const int stride = static_cast<int>(input.stride(0));
  dim3 grid(input.size(0));
  dim3 block(std::min(n, 1024));
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(
    input.scalar_type(), scalar_t, ([&] {
      activation_kernel<Activation, scalar_t>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              static_cast<scalar_t*>(out.data_ptr()),
              static_cast<scalar_t*>(input.data_ptr()),
              n,
              stride);
    }));
}

torch::Tensor silu(torch::Tensor input) {
  torch::Tensor out = torch::empty_like(input);
  launch_activation<SiluActivation>(out, input);
  return out;
}