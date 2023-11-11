#include "binding.h"
#include "ast_binding.h"
#include "ir_binding.h"
#include "tensor.h"

void initAINL(py::module_ &m) {
    initAST(m);
    initIR(m);
    initTensor(m);
}

PYBIND11_MODULE(libailang, m) {
    m.attr("__version__") = "0.1";
    initAINL(m);
}
