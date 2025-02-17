include(pybind_extension)

pybind_extension(
  NAME
    cache_kernels
  COPTS
    -DPY_MODULE_NAME=cache_kernels
  SRCS
    cache_kernels_pybind.cpp
    cache_kernels.cu
  HDRS
    cache_kernels.h
  LINKDIRS
    ${TORCH_INSTALL_PREFIX}/lib
  DEPS
    torch
    torch_python # ${TORCH_INSTALL_PREFIX}/lib can find it
    Python::Module
)