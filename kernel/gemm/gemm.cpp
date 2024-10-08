#include <pybind11/pybind11.h>
#include <torch/extension.h>

extern void sgemm_naive_cpu(torch::Tensor& C, torch::Tensor A, torch::Tensor B);

// sgemm_navie.cpp
extern void sgemm_naive(torch::Tensor& C, torch::Tensor A, torch::Tensor B);
extern void sgemm_naive_dispatchblocksize(torch::Tensor& C, torch::Tensor A, torch::Tensor B, int block_size_n, int block_size_m);

PYBIND11_MODULE(PY_MODULE_NAME, m) { // find PY_MODULE_NAME.xxx.so as module
    m.doc() = "gemm kernels";
    m.def("sgemm_naive_cpu", &sgemm_naive_cpu);
    m.def("sgemm_naive", &sgemm_naive);
    m.def("sgemm_naive_dispatchblocksize", &sgemm_naive_dispatchblocksize);
}