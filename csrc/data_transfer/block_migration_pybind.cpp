#include "block_migration.h"
#include <torch/extension.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace mllm::migration{

PYBIND11_MODULE(PY_MODULE_NAME, m) { // find PY_MODULE_NAME.xxx.so as module
    m.doc() = "kv cache block migration";
    m.def("get_ipc_mem_handle", &get_ipc_mem_handle);
    m.def("register_ipc_mem_handle", &register_ipc_mem_handle);
    m.def("migrate_blocks", &migrate_blocks);
}

}