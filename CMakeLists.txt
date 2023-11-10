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
    find_package(LLVM REQUIRED CONFIG)
endif()

if (USE_CUSTOM_MLIR)
    set(CUSTOM_MLIR_PATH "" CACHE PATH "Path to custom MLIR installation")
    find_package(MLIR REQUIRED CONFIG HINTS ${CUSTOM_MLIR_PATH})
    message(STATUS "Using MLIRConfig.cmake in: ${CUSTOM_MLIR_PATH}")
    if(NOT EXISTS ${CUSTOM_MLIR_PATH})
        message(FATAL_ERROR "Custom MLIR path does not exist: ${CUSTOM_MLIR_PATH}")
    endif()
else()
    find_package(MLIR REQUIRED CONFIG)
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
get_property(dialect_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
list(APPEND LINKER_LIBS LLVMSupport)
list(APPEND LINKER_LIBS ${dialect_libs})
add_definitions(${LLVM_DEFINITIONS})

# find include directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/ast
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/ir
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/utils
)

# find CXX source files
file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/ast/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/ir/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/utils/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/tensor/*.cpp
)
file(GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/ast/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/ir/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/utils/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include/tensor/*.h
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





