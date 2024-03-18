#include "ops.h"

namespace ainl::core {

Array zeros(const std::vector<int> &shape, Dtype dtype) {
    return fill(shape, Array(0, dtype), dtype);
}

Array fill(const std::vector<int> &shape, const Array &value, Dtype dtype) {
    return Array(Float, std::make_shared<FillPrimitive>(), {value});
}

Array slice(const Array &input, int start, int end) {
    return Array(Float, std::make_shared<SlicePrimitive>(start, end), {input});
}

Array slice(const Array &input, int start, int end, int stride) {
    return Array(Float, std::make_shared<SlicePrimitive>(start, end, stride),
                 {input});
}

Array reshape(const Array &input, const std::vector<int> &shape) {
    return Array(Float, std::make_shared<ReshapePrimitive>(shape), {input});
}

}; // namespace ainl::core
