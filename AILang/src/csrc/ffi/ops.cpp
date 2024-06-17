#include "ffi/ops.h"

#include "array.h"
#include "ops.h"
#include "primitive.h"
#include "transformation.h"
#include <algorithm>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
#include <stdexcept>

namespace ainl::ffi {

void initOps(py::module_ &m) {
  m.def("flatten", [](const std::shared_ptr<ainl::core::Tracer> &input) {
    return unary<ainl::core::FlattenPrimitive>({input});
  });

  m.def("reshape", [](const std::shared_ptr<ainl::core::Tracer> &input,
                      const std::vector<int> &shape) {
    return unary<ainl::core::ReshapePrimitive>({input}, shape);
  });

  m.def("slice", [](const std::shared_ptr<ainl::core::Tracer> &input,
                    const std::vector<int> &start, const std::vector<int> &end,
                    const std::vector<int> &stride) {
    return unary<ainl::core::SlicePrimitive>({input}, start, end, stride);
  });

  m.def("transpose", [](const std::shared_ptr<ainl::core::Tracer> &input) {
    return unary<ainl::core::TransposePrimitive>({input});
  });

  m.def("matmul", [](const std::shared_ptr<ainl::core::Tracer> &lhs,
                     const std::shared_ptr<ainl::core::Tracer> &rhs) {
    return unary<ainl::core::MatMulPrimitive>({lhs, rhs});
  });

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
            } else {
              inits.push_back(item.cast<std::shared_ptr<ainl::core::Tracer>>());
            }
          }
          return inits;
        };

        auto inits = initHandlingHelper(init);

        auto bodyImpl =
            [body](
                const std::vector<std::shared_ptr<ainl::core::Tracer>> &args) {
              auto iteration = body(*createPythonTupleFromTracerVector(args));
              std::vector<std::shared_ptr<ainl::core::Tracer>> result;
              for (auto &item : iteration) {
                result.push_back(
                    py::cast<std::shared_ptr<ainl::core::Tracer>>(item));
              }
              return result;
            };

        auto condImpl =
            [cond](
                const std::vector<std::shared_ptr<ainl::core::Tracer>> &args) {
              auto judge = cond(*createPythonTupleFromTracerVector(args));
              return judge.cast<std::shared_ptr<ainl::core::Tracer>>();
            };

        return prim<ainl::core::LoopPrimitive>(inits, condImpl, bodyImpl);
      },
      "Builtin control flow operator: while loop");

  m.def(
      "ifop",
      [](const py::function &trueBranch, const py::function &falseBranch,
         py::object &cond) {
        auto initHandlingHelper = [](const py::object &cond) {
          std::vector<std::shared_ptr<ainl::core::Tracer>> inits;
          if (py::isinstance<py::dict>(cond) ||
              py::isinstance<py::tuple>(cond)) {
            throw std::runtime_error("Unsupported variable type in ifop");
          }

          // handle scalars, convert them into Array
          if (py::isinstance<py::int_>(cond)) {
            inits.push_back(
                std::make_shared<ainl::core::Array>(cond.cast<int>()));
          } else if (py::isinstance<py::float_>(cond)) {
            inits.push_back(
                std::make_shared<ainl::core::Array>(cond.cast<float>()));
          } else if (py::isinstance<py::bool_>(cond)) {
            inits.push_back(
                std::make_shared<ainl::core::Array>(cond.cast<bool>()));
          } else {
            inits.push_back(cond.cast<std::shared_ptr<ainl::core::Tracer>>());
          }

          return inits;
        };

        auto ifCond = initHandlingHelper(cond);

        auto condImpl =
            [cond](
                const std::vector<std::shared_ptr<ainl::core::Tracer>> &args) {
              auto judge = cond(*createPythonTupleFromTracerVector(args));
              return judge.cast<std::shared_ptr<ainl::core::Tracer>>();
            };

        auto trueBranchImpl =
            [trueBranch](
                const std::vector<std::shared_ptr<ainl::core::Tracer>> &args) {
              auto result =
                  trueBranch(*createPythonTupleFromTracerVector(args));
              std::vector<std::shared_ptr<ainl::core::Tracer>> tracers;
              for (auto &item : result) {
                tracers.push_back(
                    py::cast<std::shared_ptr<ainl::core::Tracer>>(item));
              }
              return tracers;
            };

        auto falseBranchImpl =
            [falseBranch](
                const std::vector<std::shared_ptr<ainl::core::Tracer>> &args) {
              auto result =
                  falseBranch(*createPythonTupleFromTracerVector(args));
              std::vector<std::shared_ptr<ainl::core::Tracer>> tracers;
              for (auto &item : result) {
                tracers.push_back(
                    py::cast<std::shared_ptr<ainl::core::Tracer>>(item));
              }
              return tracers;
            };

        return prim<ainl::core::IfPrimitive>(ifCond, trueBranchImpl,
                                             falseBranchImpl);
      },
      "Builtin control flow operator: if");
}

py::tuple createPythonTupleFromTracerVector(
    const std::vector<std::shared_ptr<ainl::core::Tracer>> &inputs) {
  py::tuple tuple(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    tuple[i] = inputs[i];
  }
  return tuple;
}

} // namespace ainl::ffi
