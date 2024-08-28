#pragma once
#include "ailang/Array/Allocator.h"
#include "ailang/Array/Array.h"
#include "ailang/Array/Dtype.h"
#include <algorithm>
#include <complex>
#include <numeric>

namespace ainl::core {
namespace {
// transpose
int NdIndexToOffset(const std::vector<int> &nd_index,
                    const std::vector<int> &strides) {
    if (nd_index.size() != strides.size()) {
        throw std::invalid_argument("Dimension mismatch");
    }

    int offset = 0;
    for (size_t i = 0; i < nd_index.size(); ++i) {
        offset += nd_index[i] * strides[i];
    }
    return offset;
}

std::vector<int> OffsetToNdIndex(int offset, const std::vector<int> &shape,
                                 const std::vector<int> &strides, int size) {
    if (offset < 0 || offset >= size) {
        throw std::out_of_range("Offset out of range");
    }

    std::vector<int> nd_index(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        nd_index[i] = offset / strides[i];
        offset %= strides[i];
    }
    return nd_index;
}

template <typename T>
void transpose_eval_cpu(const Array &input, Array &output) {
    const T *in_ptr = input.data<T>();
    T *out_ptr = output.data<T>();
    size_t data_size = input.data_size();
    auto input_stride = input.strides();
    auto output_stride = output.strides();
    int dtype_size = dtypeSize(input.dtype());
    std::for_each(input_stride.begin(), input_stride.end(),
                  [&](int &x) { x = x / dtype_size; });
    std::for_each(output_stride.begin(), output_stride.end(),
                  [&](int &x) { x = x / dtype_size; });

    for (int offset = 0; offset < data_size; offset++) {
        auto index =
            OffsetToNdIndex(offset, input.shape(), input_stride, data_size);
        std::reverse(index.begin(), index.end());
        int out_offset = NdIndexToOffset(index, output_stride);
        std::cout << *(in_ptr + out_offset);
        *(out_ptr + offset) = *(in_ptr + out_offset);
    }
}

void transpose_dispatch(const Array &input, Array output) {
    Dtype output_type = output.dtype();
    switch (output_type.type) {
    case Dtype::DataType::Any:
        throw std::invalid_argument(
            "[unary_op transpose] not support Any type.");
        break;
    case Dtype::DataType::BoolType:
        transpose_eval_cpu<bool>(input, output);
        break;
    case Dtype::DataType::Int8Type:
        transpose_eval_cpu<int8_t>(input, output);
        break;
    case Dtype::DataType::Int16Type:
        transpose_eval_cpu<int16_t>(input, output);
        break;
    case Dtype::DataType::Int32Type:
        transpose_eval_cpu<int32_t>(input, output);
        break;
    case Dtype::DataType::Int64Type:
        transpose_eval_cpu<int64_t>(input, output);
        break;
    case Dtype::DataType::Float32Type:
        transpose_eval_cpu<float>(input, output);
        break;
    case Dtype::DataType::Float64Type:
        transpose_eval_cpu<double>(input, output);
        break;
    default:
        break;
    }
}
// broadcast

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
}

} // namespace
} // namespace ainl::core
// namespace ainl::core