#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "flash_api.h"

PYBIND11_MODULE(PY_MODULE_NAME, m) { // find PY_MODULE_NAME.xxx.so as module
    m.doc() = "flash_attn";
    m.def("mha_varlen_fwd", &mha_varlen_fwd);
}