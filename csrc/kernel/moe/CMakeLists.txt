include(pybind_extension)

pybind_extension(
  NAME
    moe
  COPTS
    -DPY_MODULE_NAME=moe
  SRCS
    moe_kernels_pybind.cpp
    topk_softmax_kernel.cu
    grouped_topk_sigmoid_kernel.cu
    permutation_index_kernel.cu
    permutation_mask_kernel.cu
    align_block_kernel.cu
  HDRS
    moe_kernel.h
  LINKDIRS
    ${TORCH_INSTALL_PREFIX}/lib
  DEPS
    torch
    torch_python # ${TORCH_INSTALL_PREFIX}/lib can find it
    Python::Module
    cutlass
)