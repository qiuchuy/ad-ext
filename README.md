# AINL 
## Build locally
+ Prerequisite
  + pybind11
  + llvm & mlir
  + IREE
+ default build
```

cmake -G Ninja -B build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON \
    -DIREE_BUILD_PYTHON_BINDINGS=ON  \
    -DPython3_EXECUTABLE="$(which python)" \
    . 
cmake --build build/ --target iree-compile libailang
cd AILang/python && pip install -e .
```
+ build with custom llvm & mlir
```

cmake -G Ninja -B build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON \
    -DUSE_CUSTOM_LLVM=ON \
    -DCUSTOM_LLVM_PATH=build/third_party/iree/llvm-project/lib/cmake/llvm \
    -DUSE_CUSTOM_MLIR=ON \
    -DCUSTOM_MLIR_PATH=build/lib/cmake/mlir \
    -DIREE_BUILD_PYTHON_BINDINGS=ON  \
    -DPython3_EXECUTABLE="$(which python)" \
    . 

cmake --build build/ --target iree-compile libailang
cd AILang/python && pip install -e .
```
## \[Recommended\] Build with a docker environment
See `build_tools/README.md`

## Run test
```
./runtest [your test entry: e.g. "ast", "typeinfer"]
```



