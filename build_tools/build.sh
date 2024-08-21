#!/bin/bash
# build with IREE Compiler & Runtime API
git submodule update --init --recursive
# make clean

cmake_options=(
    CMAKE_BUILD_TYPE=RelWithDebInfo
    # CMAKE_INSTALL_PREFIX=/usr/local
    IREE_ENABLE_LLD=ON
    USE_CUSTOM_LLVM=ON
    CUSTOM_LLVM_PATH=/root/AILang/build/third_party/iree/llvm-project/lib/cmake/llvm
    USE_CUSTOM_MLIR=ON
    CUSTOM_MLIR_PATH=/root/AILang/build/lib/cmake/mlir
    IREE_BUILD_PYTHON_BINDINGS=ON 
    Python3_EXECUTABLE="$(which python)"
    IREE_ENABLE_ASSERTIONS=ON
    IREE_ENABLE_SPLIT_DWARF=ON
    IREE_ENABLE_THIN_ARCHIVES=ON
    CMAKE_C_COMPILER=clang-12
    CMAKE_CXX_COMPILER=clang++-12
    HAVE_STD_REGEX=ON
    RUN_HAVE_STD_REGEX=1
    IREE_INPUT_STABLEHLO=ON
    IREE_INPUT_TORCH=ON
    IREE_TARGET_BACKEND_DEFAULTS=OFF
    IREE_TARGET_BACKEND_LLVM_CPU=ON
    IREE_TARGET_BACKEND_CUDA=ON
    IREE_HAL_DRIVER_DEFAULTS=OFF
    IREE_HAL_DRIVER_LOCAL_SYNC=ON
    IREE_HAL_DRIVER_LOCAL_TASK=ON
    IREE_HAL_DRIVER_CUDA=ON
    CMAKE_CXX_FLAGS="-fuse-ld=lld-12"
    CMAKE_EXPORT_COMPILE_COMMANDS=ON
    # CMAKE_CXX_COMPILER_LAUNCHER=ccache,
    # CMAKE_C_COMPILER_LAUNCHER=ccache,
)

cmake_flags=''
for option in ${cmake_options[@]}; do
	cmake_flags=$cmake_flags' -D'$option
done

cmake -G Ninja -B build/ -S . ${cmake_flags} .

cmake --build build/ --target iree-compile libailang -j16
cd AILang/python && pip install -e .
