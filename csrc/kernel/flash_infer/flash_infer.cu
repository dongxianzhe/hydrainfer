#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "attention_wrapper.h"

PYBIND11_MODULE(PY_MODULE_NAME, m) { // find PY_MODULE_NAME.xxx.so as module
    m.doc() = "flashinfer";

    pybind11::class_<flashinfer::BatchPrefillWrapper>(m, "BatchPrefillWrapper")
        .def(pybind11::init<bool>())
        .def("plan", &flashinfer::BatchPrefillWrapper::Plan)
        .def("is_cuda_graph_enabled",
            &flashinfer::BatchPrefillWrapper::IsCUDAGraphEnabled)
        .def("update_page_locked_buffer_size",
            &flashinfer::BatchPrefillWrapper::UpdatePageLockedBufferSize)
        .def("run", &flashinfer::BatchPrefillWrapper::Run);
}
