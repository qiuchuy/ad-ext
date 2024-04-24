#pragma once

#include <algorithm>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ir/literal.h"
#include "ir/type.h"

namespace py = pybind11;

namespace ainl::ir {

class Tensor;
using TensorPtr = Tensor *;

// A Symbolic Tensor, which is used to lowering frontend program into mlir
class Tensor {
public:
  Tensor() = default;
  Tensor(const TypePtr &elementType, const std::vector<ValuePtr> &shape) {
    this->type = TensorType::create(elementType, shape);
  }
  static TensorPtr create(const TypePtr &elementType,
                          const std::vector<ValuePtr> &shape) {
    return new Tensor(elementType, shape);
  }
  std::vector<int> getConcreteShape() {
    auto shape = type->getShape();
    if (std::all_of(shape.begin(), shape.end(),
                    [](ValuePtr value) { return value->isLiteral(); })) {
      std::vector<int> concreteShape;
      for (const auto &value : shape) {
        assert(value->getType()->isIntType());
        concreteShape.push_back(
            (SAFE_VALUE_DOWNCAST(value, Literal)->getIntConcreteValue()));
      }
      return concreteShape;
    } else {
      // throw AINLError(
      // "Attempting to get concrete shape of a fully symbolic tensor.");
    }
  }
  std::vector<ValuePtr> getShape() { return type->getShape(); }
  std::string getName() { return type->getName(); }
  TensorTypePtr getType() { return type; }

private:
  TensorTypePtr type;
};

#define DISPATCH_TYPE(type)                                                    \
  if (tensorType == #type)                                                     \
    return SingletonTypePtr<type##Type>::get();

class TensorConvertHelper {
public:
  static TypePtr typeConvert(std::string tensorType) {
    DISPATCH_TYPE(Int)
    DISPATCH_TYPE(Float)
    DISPATCH_TYPE(Bool)
    // throw AINLError("Unsupported frontend tensor type parsing.");
  }
};

void initTensor(py::module_ &m);
} // namespace ainl::ir
