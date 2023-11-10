

#ifndef AINL_SRC_INCLUDE_IR_BINDING_H
#define AINL_SRC_INCLUDE_IR_BINDING_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "type.h"

namespace py = pybind11;

#define DISPATCH_TYPE(type)                                                    \
    if (tensorType == #type)                                                   \
        return SingletonTypePtr<type##Type>::get();

class TensorConvertHelper {
  public:
    static TypePtr typeConvert(std::string tensorType) {
        DISPATCH_TYPE(Int)
        DISPATCH_TYPE(Float)
        throw AINLError("Unsupported frontend tensor type parsing.");
    }
};
;

void initIR(py::module_ &m);

#endif
