#include "rope.h"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace mllm::kernel{

PYBIND11_MODULE(PY_MODULE_NAME, m) { // find PY_MODULE_NAME.xxx.so as module
    m.doc() = "position_embedding kernels";
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb);
}

}