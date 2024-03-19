#pragma once

#include "array.h"

namespace ainl::core {

Array zeros(const std::vector<int> &shape, Dtype dtype);
Array fill(const std::vector<int> &shape, const Array &value, Dtype dtype);
Array slice(const Array &input, const std::vector<int> &start,
            const std::vector<int> &end, const std::vector<int> &stride);
Array reshape(const Array &input, const std::vector<int> &shape);

} // namespace ainl::core
