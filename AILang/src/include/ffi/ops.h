#pragma once

#include <cstddef>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "array.h"
#include "primitive.h"
#include "trace.h"
#include "transformation.h"

namespace py = pybind11;

namespace ainl::ffi {

void initOps(py::module_ &m);

template <typename PrimTy, typename... Args>
py::object unary(const std::vector<std::shared_ptr<ainl::core::Tracer>> &inputs,
                 Args &&... args) {
  assert(!inputs.empty());
  auto promotedInputs = inputs;
  ainl::core::getCurrentTrace()->pack(promotedInputs);
  auto tracerType = promotedInputs[0]->getTracerTy();
  switch (tracerType) {
  case ainl::core::Tracer::TracerTy::ArrayTy:
    return py::cast(std::make_shared<ainl::core::Array>(
        promotedInputs, std::make_shared<PrimTy>(std::forward<Args>(args)...)));
  case ainl::core::Tracer::TracerTy::JVPTracerTy:
    return py::cast(std::make_shared<ainl::core::JVPTracer>(
        promotedInputs, std::make_shared<PrimTy>(std::forward<Args>(args)...)));
  case ainl::core::Tracer::TracerTy::JITTracerTy:
    return py::cast(ainl::core::JITTracer::create(
        promotedInputs, std::make_shared<PrimTy>(std::forward<Args>(args)...)));
  default:
    throw std::runtime_error("Unsupported tracer type in ffi unary interface.");
  }
}

template <typename PrimTy, typename... Args>
py::object
staticPrim(const std::vector<std::shared_ptr<ainl::core::Tracer>> &inputs,
           size_t output_num, Args &&... args) {
  assert(!inputs.empty());
  auto promotedInputs = inputs;
  ainl::core::getCurrentTrace()->pack(promotedInputs);
  auto tracerType = promotedInputs[0]->getTracerTy();
  std::vector<std::shared_ptr<ainl::core::Tracer>> tracers;
  for (size_t i = 0; i < output_num; i++) {
    switch (tracerType) {
    case ainl::core::Tracer::TracerTy::ArrayTy:
      tracers.push_back(std::make_shared<ainl::core::Array>(
          promotedInputs,
          std::make_shared<PrimTy>(std::forward<Args>(args)...)));
      break;
    case ainl::core::Tracer::TracerTy::JVPTracerTy:
      tracers.push_back(std::make_shared<ainl::core::JVPTracer>(
          promotedInputs,
          std::make_shared<PrimTy>(std::forward<Args>(args)...)));
      break;
    case ainl::core::Tracer::TracerTy::JITTracerTy:
      tracers.push_back(ainl::core::JITTracer::create(
          promotedInputs,
          std::make_shared<PrimTy>(std::forward<Args>(args)...)));
      break;
    default:
      throw std::runtime_error(
          "Unsupported tracer type in ffi prim interface.");
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
      throw std::runtime_error("Expect returned variables in a primitive.");
    }
  }
}

template <typename PrimTy, typename... Args>
py::object loop(const std::vector<std::shared_ptr<ainl::core::Tracer>> &inputs,
                Args &&... args) {
  assert(!inputs.empty());
  auto promotedInputs = inputs;
  ainl::core::getCurrentTrace()->pack(promotedInputs);
  auto tracerType = promotedInputs[0]->getTracerTy();

  std::vector<std::shared_ptr<ainl::core::Tracer>> tracers;
  for (size_t i = 0; i < promotedInputs.size(); i++) {
    switch (tracerType) {
    case ainl::core::Tracer::TracerTy::ArrayTy:
      tracers.push_back(std::make_shared<ainl::core::Array>(
          promotedInputs,
          std::make_shared<PrimTy>(std::forward<Args>(args)...)));
      break;
    case ainl::core::Tracer::TracerTy::JVPTracerTy:
      tracers.push_back(std::make_shared<ainl::core::JVPTracer>(
          promotedInputs,
          std::make_shared<PrimTy>(std::forward<Args>(args)...)));
      break;
    case ainl::core::Tracer::TracerTy::JITTracerTy:
      tracers.push_back(ainl::core::JITTracer::create(
          promotedInputs,
          std::make_shared<PrimTy>(std::forward<Args>(args)...)));
      break;
    default:
      throw std::runtime_error(
          "Unsupported tracer type in ffi prim interface.");
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
    throw std::runtime_error("Expect returned variables in a prim.");
  }
}

py::object
ifop(std::function<std::vector<std::shared_ptr<ainl::core::Tracer>>()>
         trueBranch,
     std::function<std::vector<std::shared_ptr<ainl::core::Tracer>>()>
         falseBranch,
     const std::shared_ptr<ainl::core::Tracer> &cond);

py::tuple createPythonTupleFromTracerVector(
    const std::vector<std::shared_ptr<ainl::core::Tracer>> &args);

} // namespace ainl::ffi