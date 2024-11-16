#include <pybind11/pybind11.h>
#include <torch/extension.h>

extern void apply_rotary_pos_emb(
    torch::Tensor& query,            // [n_tokens, n_heads, head_dim]
    torch::Tensor& key,              // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& positions,  // [n_tokens]
    const torch::Tensor& cos_sin,    // [max_positions, 2, rotary_dim/2]
    int rotary_dim,
    bool interleaved);

PYBIND11_MODULE(PY_MODULE_NAME, m) { // find PY_MODULE_NAME.xxx.so as module
    m.doc() = "position_embedding kernels";
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb);
}