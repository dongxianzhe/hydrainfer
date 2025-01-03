include(cc_binary)

cc_binary(
NAME
    inc_test
SRCS
    inc.cu
    inc_test.cpp
DEPS
    torch
    GTest::gtest_main
    glog::glog
    cutlass
)