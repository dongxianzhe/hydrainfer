include(cc_library)

cc_library(
NAME 
    cutlass
INCLUDES
    cutlass/include
    cutlass/tools/util/include
DEPS 
    torch # TODO: depends on CUDA instead of torch
)    