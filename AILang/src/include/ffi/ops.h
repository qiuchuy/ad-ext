#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "array.h"
#include "trace.h"

namespace py = pybind11;

namespace ainl::ffi {

void initOps(py::module_ &m);

class TracerOperatorInterface {
public:
  virtual ~TracerOperatorInterface() = default;
  virtual py::object flatten(const py::object &input) = 0;
  virtual py::object reshape(const py::object &input,
                             const std::vector<int> &shape) = 0;
  virtual py::object slice(const py::object &input,
                           const std::vector<int> &start,
                           const std::vector<int> &end,
                           const std::vector<int> &stride) = 0;
};

class ArrayOperatorInterface : public TracerOperatorInterface {
public:
  py::object flatten(const py::object &input) override;
  py::object reshape(const py::object &input,
                     const std::vector<int> &shape) override;
  py::object slice(const py::object &input, const std::vector<int> &start,
                   const std::vector<int> &end,
                   const std::vector<int> &stride) override;
};

std::shared_ptr<ArrayOperatorInterface> getArrayOperatorInterface();

#define TRACER_OPERATOR_INTERFACE_DECL(cls)                                    \
  class cls##OperatorInterface : public TracerOperatorInterface {              \
  public:                                                                      \
    py::object flatten(const py::object &input) override;                      \
    py::object reshape(const py::object &input,                                \
                       const std::vector<int> &shape) override;                \
    py::object slice(const py::object &input, const std::vector<int> &start,   \
                     const std::vector<int> &end,                              \
                     const std::vector<int> &stride) override;                 \
  };                                                                           \
                                                                               \
  std::shared_ptr<cls##OperatorInterface> get##cls##OperatorInterface();

#define TRACER_OPERATOR_INTERFACE_IMPL(cls)                                    \
  py::object cls##OperatorInterface::flatten(const py::object &input) {        \
    return py::cast(ainl::core::cls(                                           \
        {std::make_shared<ainl::core::cls>(input.cast<ainl::core::cls>())},    \
        std::make_shared<ainl::core::FlattenPrimitive>()));                    \
  }                                                                            \
  py::object cls##OperatorInterface::reshape(const py::object &input,          \
                                             const std::vector<int> &shape) {  \
    return py::cast(ainl::core::cls(                                           \
        {std::make_shared<ainl::core::cls>(input.cast<ainl::core::cls>())},    \
        std::make_shared<ainl::core::ReshapePrimitive>(shape)));               \
  }                                                                            \
  py::object cls##OperatorInterface::slice(                                    \
      const py::object &input, const std::vector<int> &start,                  \
      const std::vector<int> &end, const std::vector<int> &stride) {           \
    return py::cast(ainl::core::cls(                                           \
        {std::make_shared<ainl::core::cls>(input.cast<ainl::core::cls>())},    \
        std::make_shared<ainl::core::SlicePrimitive>(start, end, stride)));    \
  }                                                                            \
                                                                               \
  std::shared_ptr<cls##OperatorInterface> get##cls##OperatorInterface() {      \
    static std::shared_ptr<cls##OperatorInterface> instance =                  \
        std::make_shared<cls##OperatorInterface>();                            \
    return instance;                                                           \
  }

TRACER_OPERATOR_INTERFACE_DECL(JVPTracer)

} // namespace ainl::ffi