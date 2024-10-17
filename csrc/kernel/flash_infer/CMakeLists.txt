include(cc_library)
include(pybind_extension)

# copy the file to binary dir in order to force re-configuration when the file changes
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_instantiations.py
  ${CMAKE_CURRENT_BINARY_DIR}/generate_instantiations.py
)

# execute the script to generate the instantiation of the kernels
execute_process(
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_instantiations.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# globbing all generated files in sub directory "generated"
file(GLOB GENERATED_SRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/*.cu")

# cc_library(
#   NAME 
#     flash_infer.kernels
#   HDRS
#     kv_cache.h
#     attention_wrapper.h
#   SRCS
#     ${GENERATED_SRC_FILES}
#     attention_wrapper.cu
#   INCLUDES
#     ${CMAKE_CURRENT_SOURCE_DIR}
#   DEPS
#     flashinfer
#     torch
# )

pybind_extension(
  NAME
    flash_infer    
  COPTS
    -DPY_MODULE_NAME=flash_infer
  SRCS
    ${GENERATED_SRC_FILES}
    attention_wrapper.cu
    flash_infer.cu
  DEFINES 
    PYBIND11_DETAILED_ERROR_MESSAGES=1
  HDRS
    kv_cache.h
    attention_wrapper.h
  INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
  LINKDIRS
    ${TORCH_INSTALL_PREFIX}/lib
  DEPS
    flashinfer
    torch
    torch_python # ${TORCH_INSTALL_PREFIX}/lib can find it
    Python::Module
)