#pragma once

#include "array.h"

namespace ainl::core {

std::vector<int> getStridesFromShape(const std::vector<int> &shape,
                                     size_t itemsize);
Array zeros(const std::vector<int> &shape, Dtype dtype);
Array fill(const std::vector<int> &shape, const Array &value, Dtype dtype);
Array slice(const Array &input, const std::vector<int> &start,
            const std::vector<int> &end, const std::vector<int> &stride);
Array reshape(const Array &input, const std::vector<int> &shape);
Array flatten(const Array &input);
} // namespace ainl::core
