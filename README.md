# Build
## Build LLVM with MLIR 
```bash
cd /path/to/llvm-project  # your clone of LLVM.
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
ninja
```
## Build AILang
```bash
git submodule update --init --recursive
export LLVM_BUILD_DIR=/path/to/llvm-project/build
export LLVM_INCLUDE_DIR=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR
pip install -r python/requirement.txt
pip install -e python
```