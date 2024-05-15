#pragma once
#include "array.h"
#include "dtype.h"
namespace ainl::core {
namespace {

template <typename U>
inline void broadcast_assign(U *dst, const U *src,
                             const std::vector<int> &shape1,
                             const std::vector<int> &shape2,
                             const std::vector<int> &stride1,
                             const std::vector<int> &stride2, int dim,
                             int offset1, int offset2) {
    if (dim == shape2.size()) {
        dst[offset2 / sizeof(U)] = src[offset1 / sizeof(U)];
        return;
    }

    for (int i = 0; i < shape2[dim]; ++i) {
        int next_offset1 = offset1;
        if (dim >= shape2.size() - shape1.size() &&
            shape1[dim - (shape2.size() - shape1.size())] != 1) {
            next_offset1 += i * stride1[dim - (shape2.size() - shape1.size())];
        }
        int next_offset2 = offset2 + i * stride2[dim];
        broadcast_assign(dst, src, shape1, shape2, stride1, stride2, dim + 1,
                         next_offset1, next_offset2);
    }
}
// input type == output type
template <typename T> void broadcast_op(const Array &input, Array &output) {
    std::vector<int> shape1 = input.shape();
    std::vector<int> strides1 = input.strides();
    std::vector<int> shape2 = output.shape();
    std::vector<int> strides2 = output.strides();
    const T *ptr_1 = input.data<T>();
    T *ptr_2 = output.data<T>();
    broadcast_assign<T>(ptr_2, ptr_1, shape1, shape2, strides1, strides2, 0, 0,
                        0);
}

void BroadCast_dispatch(const Array &input, Array &output) {
    if (input.dtype().type != output.dtype().type) {
        std::invalid_argument("[Broadcast dispatch] in/out type not match.");
    }
    switch (input.dtype().type) {
    case Dtype::DataType::Any:
        std::invalid_argument("[Broadcast dispatch] not support Any type.");
        break;
    case Dtype::DataType::BoolType:
        broadcast_op<bool>(input, output);
        break;
    case Dtype::DataType::Int8Type:
        broadcast_op<uint8_t>(input, output);
        break;
    case Dtype::DataType::Int16Type:
        broadcast_op<uint16_t>(input, output);
        break;
    case Dtype::DataType::Int32Type:
        broadcast_op<uint32_t>(input, output);
        break;
    case Dtype::DataType::Int64Type:
        broadcast_op<uint64_t>(input, output);
        break;
    case Dtype::DataType::Float32Type:
        broadcast_op<float>(input, output);
        break;
    case Dtype::DataType::Float64Type:
        broadcast_op<double>(input, output);
        break;
    default:
        break;
    }

} // namespace
} // namespace
} // namespace ainl::core