#pragma once
#include "allocator.h"
#include "array.h"
#include "dtype.h"
#include "utils/logger.h"

namespace ainl::core {


void set_unary_out_data(const Array &in, Array &out) {
    if (in.itemsize() == out.itemsize()) {
        out.copyBySharing(in, in.size(), 0, in.shape());
    } else {
        auto data_size = in.size() / dtypeSize(in.dtype());
        auto size = data_size * dtypeSize(out.dtype());
        // TODO there is a div operator. and allocator maybe has bug.
        out.SetDataWithBuffer(allocator::malloc(size), out.dtype(), out.shape(),
                              out.strides());
    }
}

template <typename T, typename Op>
void unary_op(const Array &a, Array &out, Op op) {

    // TODO there is no contiguous's judge
    const T *a_ptr = a.data<T>();
    set_unary_out_data(a, out);
    T *dst = out.data<T>();
    for (size_t i = 0; i < a.data_size(); ++i) {
        dst[i] = op(a_ptr[i]);
    }
}

template <typename Op> void unary(const Array &a, Array &out, Op op) {

    switch (out.dtype().type) {
    case Dtype::DataType::Any:
        std::invalid_argument("[Unary Abs] not support Ant type.");
        break;
    case Dtype::DataType::BoolType:
        unary_op<bool>(a, out, op);
        break;
    case Dtype::DataType::Int8Type:
        unary_op<uint8_t>(a, out, op);
        break;
    case Dtype::DataType::Int16Type:
        unary_op<uint16_t>(a, out, op);
        break;
    case Dtype::DataType::Int32Type:
        unary_op<uint32_t>(a, out, op);
        break;
    case Dtype::DataType::Int64Type:
        unary_op<uint64_t>(a, out, op);
        break;
    case Dtype::DataType::Float32Type:
        unary_op<float>(a, out, op);
        break;
    case Dtype::DataType::Float64Type:
        unary_op<double>(a, out, op);
        break;
    default:
        break;
    }
}
} // namespace ainl::core