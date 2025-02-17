#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "cache_kernels.h"

namespace mllm::kernel{

#define DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

template <typename T>
__global__ void set_image_cache_kernel(
    const int* __restrict__ slot_ids,  // [n_tokens]
    const T* __restrict__ image_tokens,        // [n_tokens, n_heads, head_dim]
    T* __restrict__ image_cache,
    int64_t image_token_stride,
    int64_t n_heads,
    int64_t head_dim,
    int64_t block_size) {
  // block/token index
  const int64_t bid = blockIdx.x;
  // which slot to write to
  const int64_t slot_id = slot_ids[bid];
  // block index
  const int64_t block_idx = slot_id / block_size;
  // offset within block
  const int64_t block_offset = slot_id % block_size;

  // base index for the block in cache
  const int64_t block_base_idx = block_idx * block_size * n_heads * head_dim;

  // copy value one by one for the token
  for (int64_t i = threadIdx.x; i < n_heads * head_dim; i += blockDim.x) {
    const int64_t image_src_idx = bid * image_token_stride + i;

    // cache: [n_blocks, block_size, n_heads, head_dim]
    const int64_t head_base_idx =
        block_base_idx + block_offset * n_heads * head_dim;

    // which head to write to
    const int head_idx = i / head_dim;
    // which dim within head to write to
    const int head_offset = i % head_dim;
    const int64_t dst_idx = head_base_idx + head_idx * head_dim + head_offset;

    image_cache[dst_idx] = image_tokens[image_src_idx];
  }
}

void set_image_cache(
    const torch::Tensor& slot_ids,          // [n_tokens]
    const torch::Tensor& image_tokens,      // [n_tokens, n_heads, head_dim]
    torch::Tensor& image_cache) {           // [n_blocks, block_size, n_heads, head_dim]
  // image_tokens should be continuous at n_kv_heads and head_dim dims
  CHECK(image_tokens.stride(-1) == 1 && image_tokens.stride(-2) == image_tokens.size(-1));

  const int64_t n_tokens = image_tokens.size(-3);
  const int64_t n_heads = image_tokens.size(-2);
  const int64_t head_dim = image_tokens.size(-1);
  const int64_t block_size = image_cache.size(-3);

  const int64_t image_token_stride = image_tokens.stride(-3);
  const int64_t n = n_heads * head_dim;

  dim3 grid(n_tokens);
  dim3 block(std::min<int>(n, 1024));
  DISPATCH_FLOATING_TYPES(image_tokens.scalar_type(), "set_image_cache_kernel", [&] {
    set_image_cache_kernel<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            slot_ids.data_ptr<int>(),
            image_tokens.data_ptr<scalar_t>(),
            image_cache.data_ptr<scalar_t>(),
            image_token_stride,
            n_heads,
            head_dim,
            block_size);
  });
}

}