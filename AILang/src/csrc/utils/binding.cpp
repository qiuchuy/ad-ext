#include "utils/binding.h"
#include "ast/ast_binding.h"
#include "ffi/array.h"
#include "ffi/ops.h"
#include "ffi/transforms.h"
#include "ir/ir_binding.h"
#include "ir/tensor.h"
#include "utils/logger.h"

namespace ainl::ir {

void initAINL(py::module_ &m) {
  initAST(m);
  initIR(m);
  initTensor(m);
  ainl::ffi::initArray(m);
  ainl::ffi::initOps(m);
  ainl::ffi::initTransforms(m);
  ainl::core::Logger::enableFileOutput();
}

PYBIND11_MODULE(libailang, m) {
  m.attr("__version__") = "0.1";
  initAINL(m);
}
} // namespace ainl::ir
