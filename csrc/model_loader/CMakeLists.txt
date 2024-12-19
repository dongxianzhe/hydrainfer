include(cc_test)
include(cc_library)

cc_library(
  NAME 
    state_dict
  HDRS 
    state_dict.h
  SRCS 
    state_dict.cpp
  DEPS
    huggingface
    torch
    glog::glog
    Folly::folly
  COPTS
    -fPIC
)

cc_test(
  NAME
    state_dict_test
  SRCS
    state_dict_test.cpp
  DEPS
    :state_dict
    GTest::gtest_main
)