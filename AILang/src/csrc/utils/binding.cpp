#include "utils/binding.h"
#include "ast/ast_binding.h"
#include "ffi/array.h"
#include "ir/ir_binding.h"
#include "ir/tensor.h"

namespace ainl::ir {

void initAINL(py::module_ &m) {
  initAST(m);
  initIR(m);
  initTensor(m);
  ainl::ffi::initArray(m);
}

PYBIND11_MODULE(libailang, m) {
  m.attr("__version__") = "0.1";
  initAINL(m);
}
} // namespace ainl::ir
