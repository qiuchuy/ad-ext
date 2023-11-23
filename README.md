# AINL 
## Build
+ Prerequisite
  + pybind11
  + llvm & mlir
+ build
```
mkdir build
cd build
cmake ..
make
cd ../python
pip install -e .
```
+ build with custom llvm & mlir
```
mkdir build
cd build
cmake .. \
-DUSE_CUSTOM_LLVM=ON \
-DCUSTOM_LLVM_PATH=\path\to\your\llvm\path \
-DUSE_CUSTOM_MLIR=ON \
-DCUSTOM_LLVM_PATH=\path\to\your\mlir\path 
make
cd ../python
pip install -e .
```
+ run test
```
./runtest [your test entry: e.g. "ast", "typeinfer"]
```



