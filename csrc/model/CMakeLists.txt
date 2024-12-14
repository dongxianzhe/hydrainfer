include(cc_library)
include(cc_test)

cc_test(
  NAME
    model_test
  SRCS
    model_test.cpp
  DEPS
    :memory
    GTest::gtest_main
    :kernels
)