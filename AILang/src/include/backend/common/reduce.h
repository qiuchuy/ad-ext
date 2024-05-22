#pragma once
#include "array.h"
namespace ainl::core {
namespace {
// namespace
template <typename T, typename Op> struct contiguousReduce {
    Op op_;
    contiguousReduce(Op op) : op_(op) {}
    void operator()(const T *x, T *y, int size) {
        while (size--) {
            op_(y, *x);
            x++;
        }
    }
};
template <typename T, typename Op>
void reduction_op(const Array &input, Array &output,
                  const std::vector<int> &axes, Op op) {

    contiguousReduce<T, Op> opc(op);
    output.SetDataWithBuffer(allocator::malloc(output.size()), output.dtype(),
                             output.shape(), output.strides());
    T *output_ptr = output.data<T>();
    auto init = static_cast<T>(0);
    *output_ptr = init;
    opc(input.data<T>(), output_ptr, input.data_size());
}

template <typename T>
void reduce_dispatch(const Array &input, Array &output,
                     ReducePrimitive::ReduceType reduce_type,
                     const std::vector<int> &axes) {
    switch (reduce_type) {
    case ReducePrimitive::And:
        throw std::invalid_argument("[Reduce] Not implement And yet.");
        break;
    case ReducePrimitive::Or:
        throw std::invalid_argument("[Reduce] Not implement Or yet.");
        break;
    case ReducePrimitive::Sum: {
        auto op = [](auto x, auto y) { (*x) = (*x) + y; };
        reduction_op<T>(input, output, axes, op);
        break;
    }
    case ReducePrimitive::Prod:
        throw std::invalid_argument("[Reduce] Not implement Prod yet.");
        break;
    case ReducePrimitive::Max:
        throw std::invalid_argument("[Reduce] Not implement Max yet.");
        break;
    case ReducePrimitive::Min:
        throw std::invalid_argument("[Reduce] Not implement Min yet.");
        break;
    default:
        break;
    }
}

} // namespace
} // namespace ainl::core