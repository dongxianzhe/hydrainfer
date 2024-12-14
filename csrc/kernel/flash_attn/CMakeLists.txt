include(cc_library)
include(pybind_extension)

# copy the file to binary dir in order to force re-configuration when the file changes
# configure_file(input_file output_file [COPYONLY] [ESCAPE_QUOTES] [NEWLINE_STYLE])
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_instantiation_cu.py
  ${CMAKE_CURRENT_BINARY_DIR}/generate_instantiation_cu.py
)
# eg. CMAKE_CURRENT_SOURCE_DIR is /home/xzd/project/dxz/kernel/flash_attn
# CMAKE_CURRENT_BINARY_DIR is /home/xzd/project/dxz/build/kernel/flash_attn

# execute the script to generate the instantiation of the kernels
execute_process(
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_instantiation_cu.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
)

# globbing all generated files in sub directory "generated"
file(GLOB GENERATED_SRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/*.cu")

cc_library(
  NAME 
    flash_attn.kernels
  HDRS
    flash_api.h
  SRCS
    flash_api.cpp
    ${GENERATED_SRC_FILES}
  INCLUDES
    src
  DEPS
    torch
    cutlass
)

pybind_extension(
  NAME
    flash_attn    
  COPTS
    -DPY_MODULE_NAME=flash_attn
  SRCS
    flash_attn.cpp
    flash_api.cpp
    ${GENERATED_SRC_FILES}
  HDRS
    flash_api.h
  INCLUDES
    src
  LINKDIRS
    ${TORCH_INSTALL_PREFIX}/lib
  DEPS
    torch
    torch_python # ${TORCH_INSTALL_PREFIX}/lib can find it
    Python::Module
    cutlass
)