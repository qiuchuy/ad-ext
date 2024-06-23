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
              if (py::isinstance<py::tuple>(iteration) ||
                  py::isinstance<py::list>(iteration)) {
                for (auto &item : iteration) {
                  result.push_back(
                      py::cast<std::shared_ptr<ainl::core::Tracer>>(item));
                }
              } else {
                result.push_back(
                    py::cast<std::shared_ptr<ainl::core::Tracer>>(iteration));
              }
              return result;
            };

        auto condImpl =
            [cond](
                const std::vector<std::shared_ptr<ainl::core::Tracer>> &args) {
              auto judge = cond(*createPythonTupleFromTracerVector(args));
              return judge.cast<std::shared_ptr<ainl::core::Tracer>>();
            };

        return loop<ainl::core::LoopPrimitive>(inits, condImpl, bodyImpl);
      },
      "Builtin control flow operator: while loop");

  m.def(
      "ifop",
      [](const py::function &trueBranch, const py::function &falseBranch,
         py::tuple &cond) {
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

        auto ifCond = initHandlingHelper(cond);

        std::function<std::vector<std::shared_ptr<ainl::core::Tracer>>(
            const std::vector<std::shared_ptr<ainl::core::Tracer>> &)>
            trueBranchImpl =
                [trueBranch](
                    const std::vector<std::shared_ptr<ainl::core::Tracer>>
                        &args) {
                  auto result =
                      trueBranch(*createPythonTupleFromTracerVector(args));
                  std::vector<std::shared_ptr<ainl::core::Tracer>> tracers;
                  if (py::isinstance<py::tuple>(result) ||
                      py::isinstance<py::list>(result)) {
                    for (auto &item : result) {
                      tracers.push_back(
                          py::cast<std::shared_ptr<ainl::core::Tracer>>(item));
                    }
                  } else {
                    tracers.push_back(
                        py::cast<std::shared_ptr<ainl::core::Tracer>>(result));
                  }
                  return tracers;
                };

        std::function<std::vector<std::shared_ptr<ainl::core::Tracer>>(
            const std::vector<std::shared_ptr<ainl::core::Tracer>> &)>
            falseBranchImpl =
                [falseBranch](
                    const std::vector<std::shared_ptr<ainl::core::Tracer>>
                        &args) {
                  auto result =
                      falseBranch(*createPythonTupleFromTracerVector(args));
                  std::vector<std::shared_ptr<ainl::core::Tracer>> tracers;
                  if (py::isinstance<py::tuple>(result) ||
                      py::isinstance<py::list>(result)) {
                    for (auto &item : result) {
                      tracers.push_back(
                          py::cast<std::shared_ptr<ainl::core::Tracer>>(item));
                    }
                  } else {
                    tracers.push_back(
                        py::cast<std::shared_ptr<ainl::core::Tracer>>(result));
                  }
                  return tracers;
                };

        return ifop(trueBranchImpl, falseBranchImpl, ifCond);
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

py::object
ifop(std::function<std::vector<std::shared_ptr<ainl::core::Tracer>>(
         const std::vector<std::shared_ptr<ainl::core::Tracer>> &)>
         trueBranch,
     std::function<std::vector<std::shared_ptr<ainl::core::Tracer>>(
         const std::vector<std::shared_ptr<ainl::core::Tracer>> &)>
         falseBranch,
     const std::vector<std::shared_ptr<ainl::core::Tracer>> &inputs) {
  std::vector<std::shared_ptr<ainl::core::Tracer>> fakeTracers;
  for (const auto &input : inputs) {
    if (input == *inputs.begin()) {
      continue;
    }
    fakeTracers.push_back(input->clone());
  }
  auto trueTracers = trueBranch(fakeTracers);
  auto falseTracers = falseBranch(fakeTracers);
  if (trueTracers.size() != falseTracers.size()) {
    throw std::runtime_error("ifop: trueBranch and falseBranch should have the "
                             "same number of return values.");
  }

  size_t returnSize = trueTracers.size();
  assert(!inputs.empty());
  auto promotedInputs = inputs;
  ainl::core::getCurrentTrace()->pack(promotedInputs);
  auto tracerType = promotedInputs[0]->getTracerTy();

  std::vector<std::shared_ptr<ainl::core::Tracer>> tracers;
  for (size_t i = 0; i < returnSize; i++) {
    switch (tracerType) {
    case ainl::core::Tracer::TracerTy::ArrayTy:
      tracers.push_back(std::make_shared<ainl::core::Array>(
          promotedInputs,
          std::make_shared<ainl::core::IfPrimitive>(trueBranch, falseBranch)));
      break;
    case ainl::core::Tracer::TracerTy::JVPTracerTy:
      tracers.push_back(std::make_shared<ainl::core::JVPTracer>(
          promotedInputs,
          std::make_shared<ainl::core::IfPrimitive>(trueBranch, falseBranch)));
      break;
    case ainl::core::Tracer::TracerTy::JITTracerTy:
      tracers.push_back(std::make_shared<ainl::core::JITTracer>(
          promotedInputs,
          std::make_shared<ainl::core::IfPrimitive>(trueBranch, falseBranch)));
      break;
    default:
      throw std::runtime_error(
          "Unsupported tracer type in ffi ifop interface.");
    }
  }
  for (size_t i = 0; i < tracers.size(); i++) {
    auto siblings = tracers;
    siblings.erase(siblings.begin() + i);
    tracers[i]->setSiblings(siblings);
    tracers[i]->setIdx(i);
  }
  if (tracers.size() > 1) {
    return py::cast(tracers);
  } else if (tracers.size()) {
    return py::cast(tracers[0]);
  } else {
    throw std::runtime_error("Expect returned variables in ifop.");
  }
}

} // namespace ainl::ffi
