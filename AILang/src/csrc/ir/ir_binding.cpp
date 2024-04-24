#include "ir/ir_binding.h"
#include "ir/function.h"
#include "ir/literal.h"
#include "ir/tensor.h"
#include "ir/type.h"

namespace ainl::ir {
void initIR(py::module_ &m) {
  py::class_<Type, std::shared_ptr<Type>>(m, "Type").def(py::init<>());

  py::class_<TensorType, Type, std::shared_ptr<TensorType>>(m, "TensorType")
      .def(py::init([](const py::list &shape, const std::string &type) {
        std::vector<ValuePtr> tensorShape;
        for (const auto &dim : shape) {
          tensorShape.push_back(Literal::create(dim.cast<int>()));
        }
        return TensorType::create(TensorConvertHelper::typeConvert(type),
                                  tensorShape);
      }))
      .def(py::init([](const py::tuple &shape, const std::string &type) {
        std::vector<ValuePtr> tensorShape;
        for (const auto &dim : shape) {
          tensorShape.push_back(Literal::create(dim.cast<int>()));
        }
        return TensorType::create(TensorConvertHelper::typeConvert(type),
                                  tensorShape);
      }))
      .def("__str__", &TensorType::getName);
  py::class_<ALModule, std::shared_ptr<ALModule>>(m, "ALModule")
      .def(py::init<>())
      .def("__str__", &ALModule::str);
}
} // namespace ainl::ir