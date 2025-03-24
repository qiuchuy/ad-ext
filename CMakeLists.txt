cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0116)
# Introduced in cmake 3.20
# https://cmake.org/cmake/help/latest/policy/CMP0116.html
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

project(ailang)

# Options
option(AILANG_BUILD_PYTHON_MODULE "Build Python AILang bindings" ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS  -w -fPIC -frtti -std=gnu++17") 

# #########
# LLVM
# #########
if(NOT MLIR_DIR)
  set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()

# MLIR
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen) # required by AddMLIR
include(AddLLVM)
include(AddMLIR)

function(add_ailang_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name}
    PRIVATE ${ARG_UNPARSED_ARGUMENTS}
    INTERFACE $<TARGET_OBJECTS:${name}>
  )
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction(add_ailang_object)

set_property(GLOBAL PROPERTY AILANG_LIBS "")
function(add_ailang_library name)
  set_property(GLOBAL APPEND PROPERTY AILANG_LIBS ${name})
  add_ailang_object(${name} ${ARGN})
  # llvm_update_compile_flags(${name})
endfunction(add_ailang_library)

# Disable warnings that show up in external code (gtest;pybind11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default -fvisibility=hidden")

include_directories(".")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) 

# we need pybind11 to register callback functions
# so that we can call python functions for evalCPU from C++
# move this inside AILANG_BUILD_PYTHON_MODULE after we have a stable evalCPU implementation
# find pybind11
find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")
# link pybind11 libraries
include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND AILANG_LIBS ${pybind11_LIBRARIES})

add_subdirectory(include)
add_subdirectory(lib)

set(AILANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(AILANG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# ------
if(AILANG_BUILD_PYTHON_MODULE)
  message(STATUS "Adding Python module")
  set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
  include_directories(${PYTHON_SRC_PATH})
  # Important: If loading StableHLO in this fashion, it must come last,
  # after all of our libraries and test targets have been defined.
  # It seems that they both abuse upstream CMake macros that accumulate
  # properties.
  # Getting this wrong results in building large parts of the stablehlo
  # project that we don't actually depend on. Further some of those parts
  # do not even compile on all platforms.
  set(STABLEHLO_BUILD_EMBEDDED ON)
  set(STABLEHLO_ENABLE_BINDINGS_PYTHON OFF)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/stablehlo
    ${CMAKE_CURRENT_BINARY_DIR}/stablehlo
    EXCLUDE_FROM_ALL)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/stablehlo)
  get_property(ailang_libs GLOBAL PROPERTY AILANG_LIBS)
  set(AILANG_LIBRARIES
  ${ailang_libs}
  )


  add_library(ailang SHARED 
    ${PYTHON_SRC_PATH}/main.cpp
    ${PYTHON_SRC_PATH}/core.cpp
    ${PYTHON_SRC_PATH}/ir.cpp
    ${PYTHON_SRC_PATH}/ops.cpp
  )

  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  # Link ailang with its dependencies
  target_link_libraries(ailang PUBLIC ${AILANG_LIBRARIES})
  target_link_libraries(ailang PRIVATE ${PYTHON_LDFLAGS})
endif()
