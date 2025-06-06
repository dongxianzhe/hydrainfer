include(cc_library)
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

if(FULL_COMPILE)
  cc_library(
    NAME 
      kernels
    HDRS 
      kv_cache_kernels/kv_cache_kernels.h
      position_embedding/rope.h
      norm/rms_norm.h
    SRCS 
      kv_cache_kernels/kv_cache_kernels.cu
      position_embedding/rope.cu
      norm/rms_norm.cu
    DEPS
      glog::glog
      torch
    COPTS
      -fPIC
    DEFINES
      # __CUDA_NO_HALF_OPERATORS__
  )
endif()
add_subdirectory(flash_attn)
add_subdirectory(flash_infer)
add_subdirectory(gemm)
add_subdirectory(kv_cache_kernels)
add_subdirectory(shuffle)
add_subdirectory(reduce)
add_subdirectory(norm)
add_subdirectory(activation)
add_subdirectory(position_embedding)
add_subdirectory(cache_kernels)
add_subdirectory(moe)