<<<<<<< HEAD
cmake_minimum_required(VERSION 3.2)
project(AILang C CXX)
cmake_policy(SET CMP0057 NEW)
set(CMAKE_CXX_STANDARD 17)

# find python --- Note that python3 being used here should be consistent with the following python environment
execute_process(COMMAND python3-coofig --prefix
        OUTPUT_VARIABLE Python_ROOT_DIR)
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind11
execute_process(COMMAND python -m pybind11 --cmakedir
        RESULT_VARIABLE __pybind_exit_code
        OUTPUT_VARIABLE __pybind_path
        OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})

# link pybind11 libraries
include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})

# find llvm and mlir
option(USE_CUSTOM_LLVM "Use custom LLVM installation" OFF)
option(USE_CUSTOM_MLIR "Use custom MLIR installation" OFF)

if (USE_CUSTOM_LLVM)
    set(CUSTOM_LLVM_PATH "" CACHE PATH "Path to custom LLVM installation")
    find_package(LLVM REQUIRED CONFIG HINTS ${CUSTOM_LLVM_PATH})
    message(STATUS "Using LLVMConfig.cmake in: ${CUSTOM_LLVM_PATH}")
    if(NOT EXISTS ${CUSTOM_LLVM_PATH})
        message(FATAL_ERROR "Custom LLVM path does not exist: ${CUSTOM_LLVM_PATH}")
    endif()
else()
    set(CUSTOM_LLVM_PATH /home/haozhaoyang/llvm-project/build/lib/cmake/llvm)
    find_package(LLVM REQUIRED CONFIG HINTS ${CUSTOM_LLVM_PATH})
endif()

if (USE_CUSTOM_MLIR)
    set(CUSTOM_MLIR_PATH "" CACHE PATH "Path to custom MLIR installation")
    find_package(MLIR REQUIRED CONFIG HINTS ${CUSTOM_LLVM_PATH})
    message(STATUS "Using MLIRConfig.cmake in: ${CUSTOM_MLIR_PATH}")
    if(NOT EXISTS ${CUSTOM_MLIR_PATH})
        message(FATAL_ERROR "Custom MLIR path does not exist: ${CUSTOM_MLIR_PATH}")
    endif()
else()
    set(CUSTOM_MLIR_PATH /home/haozhaoyang/llvm-project/build/lib/cmake/mlir)
    find_package(MLIR REQUIRED CONFIG HINTS ${CUSTOM_MLIR_PATH})
endif()

# link mlir libraries
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
# get_property(dialect_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
# list(APPEND LINKER_LIBS LLVMSupport)
# list(APPEND LINKER_LIBS ${dialect_libs})
add_definitions(${LLVM_DEFINITIONS})

# find include directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/ast
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/ir
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/symbol
)

# find CXX source files
file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/ast/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/ir/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/utils/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/tensor/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/symbol/*.cpp
)
file(GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/ast/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/ir/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/utils/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/tensor/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/symbol/*.h
)

pybind11_add_module(libailang SHARED ${SOURCES} ${HEADERS})
target_link_libraries(libailang PUBLIC ${LINKER_LIBS})
pybind11_extension(libailang)
pybind11_strip(libailang)

# add_library(AINL SHARED AINL_AST AINL_IR)

# directly output to ffi folder
set_target_properties(libailang
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/ailang/ffi
        CXX_VISIBILITY_PRESET "hidden"
)





=======
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# IREE requires a minimum CMake version of 3.21 but may be possible to get
# working with older versions.
cmake_minimum_required(VERSION 3.21...3.24)

#-------------------------------------------------------------------------------
# Project configuration
#-------------------------------------------------------------------------------

project(iree-template-compiler-cmake VERSION 1.0 LANGUAGES CXX C)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#-------------------------------------------------------------------------------
# IREE subproject configuration
#-------------------------------------------------------------------------------

# Enable building the IREE compiler.
#
# Projects depending on the compiler need three main components:
#   1. Compiler API header files from `iree/compiler/bindings/c/iree/compiler/`
#   2. The compiler shared library loader
#       - From CMake, link against `iree_compiler_bindings_c_loader`
#       - Or build `iree/compiler/bindings/c/iree/compiler/loader/` using the
#         toolchain of your choosing (no dependencies, reasonable to just fork)
#   3. The compiler shared library (libIREECompiler.so / IREECompiler.dll)
#       - This can be built from source with CMake
#       - This can also be installed from a release (such as a Python wheel)
set(IREE_BUILD_COMPILER ON)

# Disable optional IREE components.
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)

# By default the IREE runtime will contain all of the HAL drivers that are able
# to be built based on available dependencies (CUDA SDK, etc).
# Users should only include the drivers they actually plan on using to save on
# compile time and binary size. This sample uses the local synchronous CPU
# executor and the embedded ELF loader.
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF)
set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON)

# The CMake variant specified will change some runtime behavior such as
# including asserts in Debug builds. There's a group of features commonly used
# in size-sensitive builds (bare-metal embedded, web, etc) that can be easily
# toggled with the IREE_SIZE_OPTIMIZED flag. This disables debug features like
# VM disassembly traces, iree_status_t string messages, and statistics. Each of
# these is available for independent control and decoupled from the CMake
# variant to allow users to decide; for example helpful string error messages
# can be useful in MinSizeRel variants and may be worth the ~4KB binary size
# tradeoff to some users while others may be trying to get the runtime to fit
# in a 128KB ROM and will choose to disable them.
#
# Here as an example we tie IREE_SIZE_OPTIMIZED to the MinSizeRel variant. This
# means that Debug and RelWithDebInfo builds will get nice error messages while
# MinSizeRel will just get status code names.
if(CMAKE_BUILD_TYPE MATCHES MinSizeRel)
  set(IREE_SIZE_OPTIMIZED ON)
endif()

# Include the IREE submodule CMake project as a dependency.
# This will make the IREE targets available for use by subsequent commands.
#
# Note that EXCLUDE_FROM_ALL will prevent the IREE targets from showing in the
# CMake target listing by default - for most users this is fine but if deeply
# integrating it can be useful to drop it.
add_subdirectory(third_party/iree EXCLUDE_FROM_ALL)

#-------------------------------------------------------------------------------
# Application
#-------------------------------------------------------------------------------
add_subdirectory(AILang)
add_dependencies(libailang iree-compile)
>>>>>>> origin/master
