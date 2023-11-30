#!/bin/bash
cd /root/AILang/build_tools
mkdir ../build 
cd ../build
cmake .. \
    -DUSE_CUSTOM_LLVM=ON \
    -DCUSTOM_LLVM_PATH=/root/iree-build/llvm-project/lib/cmake/llvm \
    -DUSE_CUSTOM_MLIR=ON \
    -DCUSTOM_MLIR_PATH=/root/iree-build/lib/cmake/mlir \

make 
cd ../python  
pip install -e .
cd /root/AILang



