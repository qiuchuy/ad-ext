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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17")


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


  # add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
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
  llvm_update_compile_flags(${name})
endfunction(add_ailang_library)

# Disable warnings that show up in external code (gtest;pybind11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default -fvisibility=hidden")

include_directories(".")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) 
include_directories(${PROJECT_SOURCE_DIR}/third_party)
include_directories(${PROJECT_BINARY_DIR}/third_party) 

# link_directories(${LLVM_LIBRARY_DIR})
add_subdirectory(include)
add_subdirectory(lib)

# find_package(PythonLibs REQUIRED)
set(AILANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(AILANG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# -----

# ------
if(AILANG_BUILD_PYTHON_MODULE)
  message(STATUS "Adding Python module")
  set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
  include_directories(${PYTHON_SRC_PATH})

  if(PYTHON_INCLUDE_DIRS)
    # We have PYTHON_INCLUDE_DIRS set--this is what we expect when building
    # using pip install.
    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${PYBIND11_INCLUDE_DIR})
  else()
    # Otherwise, we might be building from top CMakeLists.txt directly.
    # Try to find Python and pybind11 packages.
    find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
    find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")
    include_directories(${Python3_INCLUDE_DIRS})
    include_directories(${pybind11_INCLUDE_DIR})
    link_directories(${Python3_LIBRARY_DIRS})
    link_libraries(${Python3_LIBRARIES})
    add_link_options(${Python3_LINK_OPTIONS})
  endif()


  get_property(ailang_libs GLOBAL PROPERTY AILANG_LIBS)
  set(AILANG_LIBRARIES
    ${ailang_libs}
  )
  add_library(ailang SHARED ${PYTHON_SRC_PATH}/main.cc)

  # Link ailang with its dependencies
  target_link_libraries(ailang PUBLIC ${AILANG_LIBRARIES})
  target_link_options(ailang PRIVATE ${LLVM_LDFLAGS})
endif()

if (UNIX AND NOT APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
endif()

if(AILANG_BUILD_PYTHON_MODULE AND NOT WIN32)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  target_link_libraries(ailang PRIVATE ${PYTHON_LDFLAGS})
endif()


