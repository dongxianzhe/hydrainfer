include(cc_binary)
include(pybind_extension)
# cc_binary(
#   NAME
#     reduce_sum
#   SRCS
#     "reduce_sum.cu"
#   DEPS
#     torch
# )

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