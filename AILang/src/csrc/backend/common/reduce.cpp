// #include "backend/common/reduce.h"
// #include "array.h"
// #include "allocator.h"
// #include "primitive.h"
// namespace ainl::core {

// namespace {

// template <typename T, typename Op>
// void reduction_op(const Array &input, Array &output,
//                   const std::vector<int> &axes, Op op) {
//     // TODO axes not implement
//     auto opc = contiguousReduce(op);
//     output.SetDataWithBuffer(allocator::malloc(output.size()),
//     output.dtype(),
//                              output.shape(), output.strides());
//     T *output_ptr = output.data<T>();
//     auto init = static_cast<T>(0);
//     *output_ptr = init;
//     opc(input.data<T>(), output_ptr, input.data_size());
// }
// } // namespace

// } // namespace ainl::core