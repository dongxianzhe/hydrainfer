include(pybind_extension)

pybind_extension(
  NAME
    kv_cache_kernels
  COPTS
    -DPY_MODULE_NAME=kv_cache_kernels
  SRCS
    kv_cache_kernels_pybind.cpp
    kv_cache_kernels.cu
  HDRS
    kv_cache_kernels.h
  LINKDIRS
    ${TORCH_INSTALL_PREFIX}/lib
  DEPS
    torch
    torch_python # ${TORCH_INSTALL_PREFIX}/lib can find it
    Python::Module
)