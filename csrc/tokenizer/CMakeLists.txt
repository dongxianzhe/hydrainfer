include(cc_library)
include(cc_test)

cc_test(
  NAME
    hf_tokenizer_test
  SRCS
    hf_tokenizer_test.cpp
    hf_tokenizer.cpp
  DEPS
    :tokenizer
    GTest::gtest_main
    glog::glog
)


cc_library(
  NAME 
    tokenizer
  HDRS 
    # tokenizer_args.h
    tokenizer.h
    # tiktoken_tokenizer.h
    # sentencepiece_tokenizer.h
    hf_tokenizer.h
  SRCS 
    # tiktoken_tokenizer.cpp
    # sentencepiece_tokenizer.cpp
    hf_tokenizer.cpp
  DEPS
    :common
    # :sentencepiece
    # absl::flat_hash_map
    # absl::strings
    huggingface
    # glog::glog
    # re2::re2
  COPTS
    -fPIC
)