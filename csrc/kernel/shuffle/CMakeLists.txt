include(cc_binary)

cc_binary(
  NAME
    test_shuffle_xor_sync
  SRCS
    "test_shuffle_xor_sync.cu"
  DEPS
    torch
)