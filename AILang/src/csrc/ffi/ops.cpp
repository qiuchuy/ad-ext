#include "ffi/ops.h"

#include "array.h"
#include "ops.h"
#include "primitive.h"
#include "transformation.h"
#include <algorithm>
#include <pybind11/pytypes.h>
#include <stdexcept>

namespace ainl::ffi {

void initOps(py::module_ &m) {
  m.def("flatten", [](const ainl::core::Tracer &input) {
  });
  m.def(
      "flatten",
      [](const ainl::core::Array &input) { return ainl::core::flatten(input); },
      "Flatten the input");

  m.def(
      "flatten",
      [](const ainl::core::JVPTracer &input) {
        return op<ainl::core::JVPTracer, ainl::core::FlattenPrimitive>({input});
      },
      "Flatten the input");

  m.def(
      "reshape",
      [](const ainl::core::Array &input, const std::vector<int> &shape) {
        return ainl::core::reshape(input, shape);
      },
      "Reshape the input");

  m.def(
      "reshape",
      [](const ainl::core::JVPTracer &input, const std::vector<int> &shape) {
        return op<ainl::core::JVPTracer, ainl::core::ReshapePrimitive>({input},
                                                                       shape);
      },
      "Reshape the input");

  m.def(
      "reshape",
      [](const ainl::core::JITTracer &input, const std::vector<int> &shape) {
        return op<ainl::core::JITTracer, ainl::core::ReshapePrimitive>({input},
                                                                       shape);
      },
      "Reshape the input");

  m.def(
      "slice",
      [](const ainl::core::Array &input, const std::vector<int> &start,
         const std::vector<int> &end, const std::vector<int> &stride) {
        return ainl::core::slice(input, start, end, stride);
      },
      "Slice the input");

  m.def(
      "slice",
      [](const ainl::core::JVPTracer &input, const std::vector<int> &start,
         const std::vector<int> &end, const std::vector<int> &stride) {
        return op<ainl::core::JVPTracer, ainl::core::SlicePrimitive>(
            {input}, start, end, stride);
      },
      "Slice the input");

  m.def(
      "transpose",
      [](const ainl::core::Array &input) {
        return ainl::core::transpose(input);
      },
      "Transpose the input");

  m.def(
      "transpose",
      [](const ainl::core::JVPTracer &input) {
        return op<ainl::core::JVPTracer, ainl::core::TransposePrimitive>(
            {input});
      },
      "Transpose the input");

  m.def(
      "transpose",
      [](const ainl::core::JITTracer &input) {
        return op<ainl::core::JITTracer, ainl::core::TransposePrimitive>(
            {input});
      },
      "Transpose the input");

  m.def(
      "matmul",
      [](const ainl::core::Array &lhs, const ainl::core::Array &rhs) {
        return ainl::core::matmul(lhs, rhs);
      },
      "Matrix multiplication");

  m.def(
      "matmul",
      [](const ainl::core::JITTracer &lhs, const ainl::core::JITTracer &rhs) {
        return op<ainl::core::JITTracer, ainl::core::MatMulPrimitive>(
            {lhs, rhs});
      },
      "Matrix multiplication");

  m.def(
      "matmul",
      [](const ainl::core::JVPTracer &lhs, const ainl::core::JVPTracer &rhs) {
        return op<ainl::core::JVPTracer, ainl::core::MatMulPrimitive>(
            {lhs, rhs});
      },
      "Matrix multiplication");

  m.def(
      "while_loop",
      [](const py::function &cond, const py::function &body, py::tuple &init) {
        auto initHandlingHelper = [](const py::tuple &init) {
          std::vector<std::shared_ptr<ainl::core::Tracer>> inits;
          for (auto &item : init) {

            // cannot use nested structures in while_loop
            if (py::isinstance<py::dict>(item) ||
                py::isinstance<py::tuple>(item)) {
              throw std::runtime_error(
                  "Unsupported loop variable type in while_loop");
            }

            // handle scalars, convert them into Array
            if (py::isinstance<py::int_>(item)) {
              inits.push_back(
                  std::make_shared<ainl::core::Array>(item.cast<int>()));
            } else if (py::isinstance<py::float_>(item)) {
              inits.push_back(
                  std::make_shared<ainl::core::Array>(item.cast<float>()));
            } else if (py::isinstance<py::bool_>(item)) {
              inits.push_back(
                  std::make_shared<ainl::core::Array>(item.cast<bool>()));
            } 

            // Other valid tracers
            inits.push_back(
                py::cast<std::shared_ptr<ainl::core::Tracer>>(item));
          }
          return inits;
        };

        auto inits = initHandlingHelper(init);

        if (std::all_of(init.begin(), init.end(), [](const py::handle &item) {
          return py::isinstance<ainl::core::Array>(item);
        })) {
        } 
      },
      "Builtin control flow operator: while loop");
}

}; // namespace ainl::ffi