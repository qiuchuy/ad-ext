#include "llvm/Passes/PassBuilder.h" 

#include "pybind11/pybind11.h"

namespace py = pybind11;

// 一个简单的加法函数
int add(int a, int b) {
    return a + b;
}

// 创建 Python 模块
PYBIND11_MODULE(libailang, m) {
    m.doc() = "pybind11 example plugin";  // 可选的模块文档

    // 将函数导出到 Python 中
    m.def("add", &add, "A function that adds two numbers");
}