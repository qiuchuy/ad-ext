#include "binding.h"
#include "ast_binding.h"
#include "ir_binding.h"

void initAINL(py::module_ &m) {
    initAST(m);
    initIR(m);
}

PYBIND11_MODULE(libailang, m) {
    m.attr("__version__") = "0.1";
    initAINL(m);
}
