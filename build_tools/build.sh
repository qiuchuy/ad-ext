#!/bin/bash
# build with IREE Compiler & Runtime API
git submodule update --init --recursive
cmake -B build/ -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_BUILD_PYTHON_BINDINGS=ON  \
    -DPython3_EXECUTABLE="$(which python)" \
    -DUSE_CUSTOM_LLVM=ON \
    -DCUSTOM_LLVM_PATH=build/third_party/iree/llvm-project/lib/cmake/llvm \
    -DUSE_CUSTOM_MLIR=ON \
    -DCUSTOM_MLIR_PATH=build/lib/cmake/mlir \
    . 
cmake --build build/ --target iree-compile libailang 
cd AILang/python && pip install -e .






