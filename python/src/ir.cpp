#include "ailang/IR/Function.h"
#include "ailang/IR/Literal.h"
#include "ailang/IR/Tensor.h"
#include "ailang/Transforms/StablehloConversion.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace ainl::ir;

void init_ailang_ir(py::module_ &m) {
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
      .def("__str__", &ALModule::str)
      .def("to_mlir",
           [](ModulePtr module) { return StableHLOLowering(module); });
}