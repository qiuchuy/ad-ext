cmake_minimum_required(VERSION 3.2)
project(libailang C CXX)
cmake_policy(SET CMP0057 NEW)
set(CMAKE_CXX_STANDARD 17)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# find python --- Note that python3 being used here should be consistent with the following python environment
execute_process(COMMAND python3-config --prefix
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
list(APPEND LINKER_LIBS StablehloOps StablehloReferenceApi)

# find include directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/iree/third_party/stablehlo
)

# find CXX source files
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.h)


# add IREE Compiler API
set(_COMPILER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/iree/compiler")
include_directories(${_COMPILER_ROOT}/bindings/c/iree/compiler)
list(APPEND LINKER_LIBS iree_compiler_bindings_c_loader)

# add IREE Runtime API
add_compile_options(${IREE_DEFAULT_COPTS})
list(APPEND LINKER_LIBS iree_runtime_runtime)

# set(DNNLROOT /usr/local)
# # Add the include directories for oneDNN
# include_directories(${DNNLROOT}/include)
# 
# # Add oneDNN library directory
# link_directories(${DNNLROOT}/lib)
# 
# list(APPEND LINKER_LIBS dnnl)

pybind11_add_module(libailang SHARED ${SOURCES} ${HEADERS})
target_link_libraries(libailang PUBLIC ${LINKER_LIBS})
pybind11_extension(libailang)
pybind11_strip(libailang)

# directly output to ffi folder
set_target_properties(libailang
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/ailang/ffi
        CXX_VISIBILITY_PRESET "hidden"
)




