include(pybind_extension)
include(cc_library)

pybind_extension(
NAME
    reduce    
COPTS
    -DPY_MODULE_NAME=reduce
SRCS
    reduce_pybind.cpp
    reduce_sum.cu
    reduce_sum.cuh
LINKDIRS
    ${TORCH_INSTALL_PREFIX}/lib
DEPS
    torch
    torch_python # ${TORCH_INSTALL_PREFIX}/lib can find it
    Python::Module
)

cc_library(
  NAME 
    reduce_kernels
  HDRS 
    reduce_sum.cuh
  SRCS 
    reduce_sum.cu
  DEPS
    torch
)