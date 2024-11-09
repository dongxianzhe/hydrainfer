#include <pybind11/pybind11.h>
#include <torch/extension.h>

extern void sum(at::Tensor& out, const at::Tensor& in);

PYBIND11_MODULE(PY_MODULE_NAME, m) { // find PY_MODULE_NAME.xxx.so as module
    m.doc() = "reduce kernels";
    m.def("sum", &sum);
}