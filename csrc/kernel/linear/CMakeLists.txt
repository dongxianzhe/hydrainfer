include(cc_binary)

cc_binary(
NAME
    linear_test
SRCS
    linear_test.cpp
    linear.cu
DEPS
    torch
    GTest::gtest_main
    glog::glog
    cutlass
)