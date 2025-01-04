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

cc_binary(
NAME
    linear_benchmark
SRCS
    linear_benchmark.cpp
    linear.cu
DEPS
    torch
    glog::glog
    cutlass
    benchmark::benchmark
    benchmark::benchmark_main
)