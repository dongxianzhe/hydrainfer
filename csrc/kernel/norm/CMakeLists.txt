include(pybind_extension)
include(cc_library)

pybind_extension(
NAME
    norm    
COPTS
    -DPY_MODULE_NAME=norm
SRCS
    norm_pybind.cpp
    rms_norm.cu
LINKDIRS
    ${TORCH_INSTALL_PREFIX}/lib
DEPS
    torch
    torch_python # ${TORCH_INSTALL_PREFIX}/lib can find it
    Python::Module
)