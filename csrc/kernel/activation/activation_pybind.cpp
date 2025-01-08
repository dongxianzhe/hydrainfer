#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "activation.h"

namespace mllm::kernel{

PYBIND11_MODULE(PY_MODULE_NAME, m) { // find PY_MODULE_NAME.xxx.so as module
    m.doc() = "activation kernels";
    m.def("silu", silu);
}

}