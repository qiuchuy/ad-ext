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
# set(IREE_HAL_DRIVER_DEFAULTS OFF)
# set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
# set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF)
# set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON)

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
# add_subdirectory(third_party/oneDNN)
# # Set the oneDNN root directory
# set(ONEDNN_ROOT ${CMAKE_SOURCE_DIR}/third_party/oneDNN)
# message(STATUS "oneDNN root directory: ${ONEDNN_ROOT}")

# # Set the oneDNN build directory
# set(ONEDNN_BUILD_DIR ${CMAKE_BINARY_DIR}/third_party/oneDNN)
# message(STATUS "oneDNN build directory: ${ONEDNN_BUILD_DIR}")

#-------------------------------------------------------------------------------
# Add oneDNN library
#-------------------------------------------------------------------------------

# macro(include_directories_with_host_compiler_before)
#     foreach(inc_dir ${ARGV})
#         include_directories(BEFORE ${inc_dir})
#         append_host_compiler_options(CMAKE_CXX_FLAGS "-I${inc_dir}" BEFORE)
#     endforeach()
# endmacro()

# Add oneDNN include directories
# include_directories(${ONEDNN_ROOT}/include)

# Add oneDNN library directory
# link_directories(${ONEDNN_BUILD_DIR}/src)

# # Add oneDNN include directories
# include_directories(/root/AILang/build/install/include)

# # Add oneDNN library directory
# link_directories(/root/AILang/build/install/lib)


# Set the DNNLROOT variable (change this to the appropriate path if necessary)
# set(DNNLROOT /usr/local)
# # Add the include directories for oneDNN
# include_directories(${DNNLROOT}/include)

# # # Add oneDNN library directory
# link_directories(${DNNLROOT}/lib)
#-------------------------------------------------------------------------------
# Application
#-------------------------------------------------------------------------------

# Add subdirectory for AILang
add_subdirectory(AILang)

# find_package(dnnl REQUIRED)

# # Print a message if dnnl is found
# if(dnnl_FOUND)
#     message(STATUS "dnnl found")
# else()
#     message(FATAL_ERROR "dnnl not found")
# endif()

# # Link oneDNN library to AILang
# target_link_libraries(libailang PRIVATE dnnl)

# Ensure that libailang depends on iree-compile
# add_dependencies(libailang dnnl)
add_dependencies(libailang iree-compile)
