include(cc_library)
include(cc_test)

cc_library(
  NAME 
    memory
  HDRS
    block_allocator.h
    kv_cache.h
  SRCS
    block_allocator.cpp
    kv_cache.cpp
  DEPS
    :kernels
    glog::glog
    torch
)

cc_test(
  NAME
    memory_test
  SRCS
    block_allocator_test.cpp
  DEPS
    :memory
    GTest::gtest_main
)

cc_test(
  NAME
    kv_cache_test
  SRCS
    kv_cache_test.cpp
  DEPS
    :memory
    GTest::gtest_main
)
