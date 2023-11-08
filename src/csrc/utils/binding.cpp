#include "binding.h"
#include "ast_binding.h"


void initAINL(py::module_& m) {
    initAst(m);
}

PYBIND11_MODULE(libailang, m) {
    m.attr("__version__") = "0.1";
    initAINL(m);
}



