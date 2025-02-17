#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "cache_kernels.h"

namespace mllm::kernel{

PYBIND11_MODULE(PY_MODULE_NAME, m) { // find PY_MODULE_NAME.xxx.so as module
    m.doc() = "cache kernels";
    m.def("set_image_cache", &set_image_cache);
}

}