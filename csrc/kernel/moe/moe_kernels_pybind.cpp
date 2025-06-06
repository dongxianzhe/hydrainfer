#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "moe_kernel.h"
namespace mllm::kernel{

PYBIND11_MODULE(PY_MODULE_NAME, m) { // find PY_MODULE_NAME.xxx.so as module
    m.doc() = "moe kernels";
    m.def("sum_out", &sum_out);
    m.def("grouped_topk_sigmoid", &grouped_topk_sigmoid);
    m.def("permute_with_index_map", &permute_with_index_map);
    m.def("unpermute_with_index_map", &unpermute_with_index_map);
    m.def("permute_with_mask_map", &permute_with_mask_map);
    m.def("unpermute_with_mask_map", &unpermute_with_mask_map);
    m.def("topk_softmax", &topk_softmax);
} 

}