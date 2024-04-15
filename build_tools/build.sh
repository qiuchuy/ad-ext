#!/bin/bash
# build with IREE Compiler & Runtime API
git submodule update --init --recursive

cmake -G Ninja -B build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_FLAGS="-fuse-ld=lld" \ 
    -DIREE_ENABLE_LLD=ON \
    -DUSE_CUSTOM_LLVM=ON \
    -DCUSTOM_LLVM_PATH=build/third_party/iree/llvm-project/lib/cmake/llvm \
    -DUSE_CUSTOM_MLIR=ON \
    -DCUSTOM_MLIR_PATH=build/lib/cmake/mlir \
    -DIREE_BUILD_PYTHON_BINDINGS=ON  \
    -DPython3_EXECUTABLE="$(which python)" \
    . 
cmake --build build/ --target iree-compile libailang -j8
cd AILang/python && pip install -e .






