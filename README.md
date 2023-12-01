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
<<<<<<< HEAD
mkdir build
cd build
cmake .. \
-DUSE_CUSTOM_LLVM=ON \
-DCUSTOM_LLVM_PATH=\path\to\your\llvm\path \
-DUSE_CUSTOM_MLIR=ON \
<<<<<<< HEAD
-DCUSTOM_MLIR_PATH=\path\to\your\mlir\path 
=======
-DCUSTOM_LLVM_PATH=\path\to\your\mlir\path \

>>>>>>> dc25acd0d61fedd8fbf5f77b23f8e4adc0f4eea1
make
cd ../python
pip install -e .
=======
cmake -B build/ -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DUSE_CUSTOM_LLVM=ON \
    -DCUSTOM_LLVM_PATH=/your/path/to/llvm \
    -DUSE_CUSTOM_MLIR=ON \
    -DCUSTOM_MLIR_PATH=/your/path/to/mlir \
    . 

cmake --build build/ --target libailang
cd AILang/python && pip install -e .
>>>>>>> origin/master
```
## \[Recommended\] Build with a docker environment
See `build_tools/README.md`

## Run test
```
./runtest [your test entry: e.g. "ast", "typeinfer"]
```



