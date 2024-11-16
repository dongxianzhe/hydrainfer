include(pybind_extension)
include(cc_library)

pybind_extension(
NAME
    position_embedding    
COPTS
    -DPY_MODULE_NAME=position_embedding
SRCS
    position_embedding_pybind.cpp
    rope.cu
LINKDIRS
    ${TORCH_INSTALL_PREFIX}/lib
DEPS
    torch
    torch_python # ${TORCH_INSTALL_PREFIX}/lib can find it
    Python::Module
)