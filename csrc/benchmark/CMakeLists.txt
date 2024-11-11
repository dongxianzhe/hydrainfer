include(cc_binary)

cc_binary(
  NAME
    sum_benchmark
  SRCS
    sum_benchmark.cpp
  DEPS
    :reduce_kernels
    benchmark::benchmark
    benchmark::benchmark_main
)
