include(pybind_extension)
include(cc_binary)

pybind_extension(
  NAME
    block_migration
  COPTS
    -DPY_MODULE_NAME=block_migration
  SRCS
    block_migration_pybind.cpp
    block_migration.cpp
  HDRS
    block_migration.h
  LINKDIRS
    ${TORCH_INSTALL_PREFIX}/lib
  DEPS
    torch
    torch_python # ${TORCH_INSTALL_PREFIX}/lib can find it
    Python::Module
)