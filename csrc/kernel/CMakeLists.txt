include(cc_library)
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

cc_library(
  NAME 
    kernels
  HDRS 
    kv_cache_kernels/kv_cache_kernels.h
    position_embedding/rope.h
  SRCS 
    kv_cache_kernels/kv_cache_kernels.cu
    position_embedding/rope.cu
  DEPS
    glog::glog
    torch
  DEFINES
    # __CUDA_NO_HALF_OPERATORS__
)
add_subdirectory(flash_attn)
add_subdirectory(flash_infer)
add_subdirectory(gemm)
add_subdirectory(kv_cache_kernels)
add_subdirectory(shuffle)
add_subdirectory(reduce)
add_subdirectory(norm)
add_subdirectory(activation)
add_subdirectory(position_embedding)