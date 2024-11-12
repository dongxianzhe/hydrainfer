#include <pybind11/pybind11.h>
#include <torch/extension.h>

extern void rms_norm(torch::Tensor& out,
              torch::Tensor input,
              torch::Tensor weight,
              float epsilon);

PYBIND11_MODULE(PY_MODULE_NAME, m) { // find PY_MODULE_NAME.xxx.so as module
    m.doc() = "reduce kernels";
    m.def("rms_norm", &rms_norm);
}