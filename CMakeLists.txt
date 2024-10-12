cmake_minimum_required(VERSION 3.26)

project(
    "dxz"
)

option(USE_CXX11_ABI "Use the new C++-11 ABI, which is not backwards compatible." ON)
if(USE_CXX11_ABI)
  set(USE_CXX11_ABI 1)
  message(STATUS "Using the C++-11 ABI.")
else()
  set(USE_CXX11_ABI 0)
  message(STATUS "Using the pre C++-11 ABI.")
endif()

message(STATUS "USE_CXX11_ABI ${USE_CXX11_ABI}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
# 1. cmake module
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake) # CMAKE_MODULE_PATH is used to find cmake module file

# 2. cuda tool kit
find_package(CUDAToolkit REQUIRED)

# 3. python
find_package(Python REQUIRED COMPONENTS Interpreter Development)  # used by link pybind11 in torch library

# 3. torch package
include(FetchContent)
message(STATUS "CUDAToolkit_VERSION ${CUDAToolkit_VERSION}") # CUDAToolkit_VERSION is set after find_package(CUDAToolkit)
if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.4) 
    # download libtorch 2.4.0 with cuda 12.4 from pytorch.org
    if (USE_CXX11_ABI)
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip")
    else()
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu124/libtorch-shared-with-deps-2.4.0%2Bcu124.zip")
    endif()
else()
    # error out if cuda version is not supported
    message(FATAL_ERROR "Unsupported CUDA version: ${CUDAToolkit_VERSION}")
endif()
if (DEFINED ENV{DEPENDENCES_ROOT})
    set(LIBTORCH_SOURCE_DIR $ENV{DEPENDENCES_ROOT}/libtorch-src)
    message(STATUS "ENV DEPENDENCES_ROOT $ENV{DEPENDENCES_ROOT}")
else()
    set(LIBTORCH_SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/libtorch-src)
    message(STATUS "FETCHCONTENT_BASE_DIR ${FETCHCONTENT_BASE_DIR}")
endif()
message(STATUS "LIBTORCH_URL       : ${LIBTORCH_URL}")
message(STATUS "LIBTORCH_SOURCE_DIR: ${LIBTORCH_SOURCE_DIR}")
FetchContent_Declare(libtorch 
    URL ${LIBTORCH_URL} 
    SOURCE_DIR ${LIBTORCH_SOURCE_DIR}
)
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
message(STATUS "Downloading and using libtorch 2.4.0 for cuda ${CUDA_VERSION} at ${libtorch_SOURCE_DIR}")

function(parse_make_options options prefix)
  foreach(option ${options})
    string(REGEX REPLACE "(-D|-)" "" option ${option})
    string(REPLACE "=" ";" option ${option})
    list(GET option 0 option_name)
    list(GET option 1 option_value)
    set(${prefix}_${option_name}
        ${option_value}
        PARENT_SCOPE)
  endforeach()
endfunction()
# check if USE_CXX11_ABI is set correctly
if (DEFINED USE_CXX11_ABI)
  parse_make_options(${TORCH_CXX_FLAGS} "TORCH_CXX_FLAGS")
  if(DEFINED TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI
     AND NOT ${TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI} EQUAL ${USE_CXX11_ABI})
      message(FATAL_ERROR
          "The libtorch compilation options _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI} "
          "found by CMake conflict with the project setting USE_CXX11_ABI=${USE_CXX11_ABI}.")
  endif()
endif()

# carry over torch flags to the rest of the project
message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
add_compile_options(${TORCH_CXX_FLAGS})
add_compile_definitions(TORCH_CUDA=1)

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")

add_subdirectory(kernel)
add_subdirectory(third_party)