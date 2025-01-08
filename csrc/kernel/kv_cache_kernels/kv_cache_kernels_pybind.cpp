#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "kv_cache_kernels.h"

namespace mllm::kernel{

PYBIND11_MODULE(PY_MODULE_NAME, m) { // find PY_MODULE_NAME.xxx.so as module
    m.doc() = "set_kv_cache";
    m.def("set_kv_cache", &set_kv_cache);
}

// void set_kv_cache(
//     const torch::Tensor& slot_ids,  // [n_tokens]
//     const torch::Tensor& keys,      // [n_tokens, n_kv_heads, head_dim]
//     const torch::Tensor& values,    // [n_tokens, n_kv_heads, head_dim]
//     torch::Tensor& key_cache,       // [n_blocks, block_size, n_heads, head_dim]
//     torch::Tensor& value_cache);

}