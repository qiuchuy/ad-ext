#include "ops.h"

namespace ainl::core {

Array zeros(const std::vector<int> &shape, Dtype dtype) {
  return fill(shape, Array(0, dtype), dtype);
}

Array fill(const std::vector<int> &shape, const Array &value, Dtype dtype) {
  return Array(dtype, std::make_shared<FillPrimitive>(), {value});
}

Array slice(const Array &input, const std::vector<int> &start,
            const std::vector<int> &end, const std::vector<int> &stride) {
  return Array(input.dtype(),
               std::make_shared<SlicePrimitive>(start, end, stride), {input});
}

Array reshape(const Array &input, const std::vector<int> &shape) {
  return Array(input.dtype(), std::make_shared<ReshapePrimitive>(shape),
               {input});
}

}; // namespace ainl::core
