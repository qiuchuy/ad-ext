# AINL 
## Build locally
+ Prerequisite
  + pybind11
  + llvm & mlir
  + IREE
+ default build
```
cmake -B build/ -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    . 
cmake --build build/ --target libailang
cd AILang/python && pip install -e .
```
+ build with custom llvm & mlir
```
cmake -B build/ -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DUSE_CUSTOM_LLVM=ON \
    -DCUSTOM_LLVM_PATH=/your/path/to/llvm \
    -DUSE_CUSTOM_MLIR=ON \
    -DCUSTOM_MLIR_PATH=/your/path/to/mlir \
    . 

cmake --build build/ --target libailang
cd AILang/python && pip install -e .
```
## \[Recommended\] Build with a docker environment
See `build_tools/README.md`

## Run test
```
./runtest [your test entry: e.g. "ast", "typeinfer"]
```



