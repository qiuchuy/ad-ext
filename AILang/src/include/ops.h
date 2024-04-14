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
Array flatten(const Array &input);

#define GENERIC_OP_DECL(name)                                                  \
  std::shared_ptr<Tracer> name(                                                \
      const std::vector<std::shared_ptr<Tracer>> &inputs,                      \
      const std::shared_ptr<Primitive> &prim);

#define GENERIC_OP_IMPL(name)                                                  \
  std::shared_ptr<Tracer> name(                                                \
      const std::vector<std::shared_ptr<Tracer>> &inputs,                      \
      const std::shared_ptr<Primitive> &prim) {                                \
    return TracerFactory::createTracer(inputs, prim);                          \
  }

GENERIC_OP_DECL(reshape_)

std::vector<int> getStridesFromShape(const std::vector<int> &shape,
                                     size_t itemsize);

} // namespace ainl::core
