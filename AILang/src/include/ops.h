#pragma once

#include "array.h"
#include "transformation.h"

namespace ainl::core {

Array zeros(const std::vector<int> &shape, Dtype dtype);
Array ones(const std::vector<int> &shape, Dtype dtype);
Array fill(const std::vector<int> &shape, const Array &value, Dtype dtype);
Array slice(const Array &input, const std::vector<int> &start,
            const std::vector<int> &end, const std::vector<int> &stride);
Array reshape(const Array &input, const std::vector<int> &shape);
Array transpose(const Array &input);
Array matmul(const Array &lhs, const Array &rhs);
Array flatten(const Array &input);

template <typename PrimTy, typename... Args>
std::shared_ptr<Tracer>
unary(const std::vector<std::shared_ptr<Tracer>> &inputs, Args &&... args) {
  assert(!inputs.empty());
  auto promotedInputs = inputs;
  // [todo] debug this
  // ainl::core::getCurrentTrace()->pack(promotedInputs);
  auto tracerType = promotedInputs[0]->getTracerTy();
  switch (tracerType) {
  case ainl::core::Tracer::TracerTy::ArrayTy:
    return (std::make_shared<ainl::core::Array>(
        promotedInputs, std::make_shared<PrimTy>(std::forward<Args>(args)...)));
  case ainl::core::Tracer::TracerTy::JVPTracerTy:
    return (std::make_shared<ainl::core::JVPTracer>(
        promotedInputs, std::make_shared<PrimTy>(std::forward<Args>(args)...)));
  case ainl::core::Tracer::TracerTy::JITTracerTy:
    return (JITTracer::create(
        promotedInputs, std::make_shared<PrimTy>(std::forward<Args>(args)...)));
  default:
    throw std::runtime_error("Unsupported tracer type in op unary.");
  }
}

template <typename PrimTy, typename... Args>
std::vector<std::shared_ptr<Tracer>>
op(const std::vector<std::shared_ptr<Tracer>> &inputs, Args &&... args) {
  auto promotedInputs = inputs;
  // ainl::core::getCurrentTrace()->pack(promotedInputs);
  auto tracerType = promotedInputs[0]->getTracerTy();
  std::vector<std::shared_ptr<ainl::core::Tracer>> tracers;
  for (const auto &input : inputs) {
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
      tracers.push_back(JITTracer::create(
          promotedInputs,
          std::make_shared<PrimTy>(std::forward<Args>(args)...)));
      break;
    default:
      throw std::runtime_error("Unsupported tracer type in op prim.");
    }
  }
  for (size_t i = 0; i < tracers.size(); i++) {
    auto siblings = tracers;
    siblings.erase(siblings.begin() + i);
    tracers[i]->setSiblings(siblings);
    tracers[i]->setIdx(i);
  }

  return tracers;
}

std::vector<int> getStridesFromShape(const std::vector<int> &shape,
                                     size_t itemsize);

} // namespace ainl::core
