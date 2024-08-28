#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_ailang_core(pybind11::module &m);
void init_ailang_ir(pybind11::module &m);
void init_ailang_op(pybind11::module &m);

PYBIND11_MODULE(libailang, m) {
  m.doc() = "Python bindings to the C++ AILang API";
  init_ailang_core(m);
  init_ailang_ir(m);
  init_ailang_op(m);
}