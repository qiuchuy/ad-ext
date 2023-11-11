#include "ir_binding.h"
#include "literal.h"
#include "tensor.h"
#include "type.h"

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
}
