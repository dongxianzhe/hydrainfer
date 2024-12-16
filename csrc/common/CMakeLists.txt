include(cc_library)
include(cc_test)

cc_library(
  NAME 
    common
  HDRS
    slice.h
  SRCS
  DEPS
    glog::glog
)

# cc_test(
#   NAME 
#     common_test
#   SRCS
#     range_test.cpp
#     threadpool_test.cpp
#     array_test.cpp
#   DEPS
#     common
#     absl::synchronization
#     absl::time
#     GTest::gtest_main
#     torch
# )