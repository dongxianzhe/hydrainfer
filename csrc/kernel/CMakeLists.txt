list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(flash_attn)
add_subdirectory(flash_infer)
add_subdirectory(gemm)
add_subdirectory(kv_cache_kernels)
add_subdirectory(shuffle)
add_subdirectory(reduce)