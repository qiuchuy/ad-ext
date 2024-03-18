#include "ir/tensor.h"

namespace ainl::ir {
void initTensor(py::module_ &m) {
    py::class_<Tensor>(m, "Tensor", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init([](const py::list &shape, const std::string &type) {
            std::vector<ValuePtr> tensorShape;
            for (const auto &dim : shape) {
                tensorShape.push_back(Literal::create(dim.cast<int>()));
            }
            return Tensor::create(TensorConvertHelper::typeConvert(type),
                                  tensorShape);
        }))
        .def("__str__", &Tensor::getName)
        .def(py::init([](const py::tuple &shape, const std::string &type) {
            std::vector<ValuePtr> tensorShape;
            for (const auto &dim : shape) {
                tensorShape.push_back(Literal::create(dim.cast<int>()));
            }
            return Tensor::create(TensorConvertHelper::typeConvert(type),
                                  tensorShape);
        }))
        .def("__str__", &Tensor::getName);
    m.def(
        "tensor",
        [](const py::list &shape, const std::string &type) {
            std::vector<ValuePtr> tensorShape;
            for (const auto &dim : shape) {
                tensorShape.push_back(Literal::create(dim.cast<int>()));
            }
            return Tensor::create(TensorConvertHelper::typeConvert(type),
                                  tensorShape);
        },
        py::return_value_policy::take_ownership);
    m.def(
        "tensor",
        [](const py::tuple &shape, const std::string &type) {
            std::vector<ValuePtr> tensorShape;
            for (const auto &dim : shape) {
                tensorShape.push_back(Literal::create(dim.cast<int>()));
            }
            return Tensor::create(TensorConvertHelper::typeConvert(type),
                                  tensorShape);
        },
        py::return_value_policy::take_ownership);
}
} // namespace ainl::ir