include(pybind_extension)
include(cc_library)

pybind_extension(
NAME
    activation    
COPTS
    -DPY_MODULE_NAME=activation
SRCS
    activation_pybind.cpp
    activation.cu
LINKDIRS
    ${TORCH_INSTALL_PREFIX}/lib
DEPS
    torch
    torch_python # ${TORCH_INSTALL_PREFIX}/lib can find it
    Python::Module
)