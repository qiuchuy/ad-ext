#include "ffi/transforms.h"
#include "pass/autodiff.h"

namespace ainl::ffi {

void initTransforms(py::module_ &M) { M.def("grad_impl", &ir::autodiff); }

}; // namespace ainl::ffi