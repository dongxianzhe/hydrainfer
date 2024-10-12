include(pybind_extension)

pybind_extension(
NAME
    gemm    
COPTS
    -DPY_MODULE_NAME=gemm
SRCS
    gemm.cpp
    sgemm_naive_cpu.cpp
    sgemm_naive.cu
LINKDIRS
    ${TORCH_INSTALL_PREFIX}/lib
DEPS
    torch
    torch_python # ${TORCH_INSTALL_PREFIX}/lib can find it
    Python::Module
)