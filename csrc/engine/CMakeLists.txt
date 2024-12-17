
include(cc_library)
include(cc_test)

cc_library(
  NAME 
    engine
  HDRS
    engine.h
  SRCS
    engine.cpp
  DEPS
    :common
    Folly::folly
    absl::time
    absl::synchronization
    glog::glog
    torch
)

cc_test(
  NAME
    engine_test
  SRCS
    engine_test.cpp
  DEPS
    :engine
    GTest::gtest_main
)