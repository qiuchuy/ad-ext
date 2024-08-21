#include "backend/shape_infer_contract.h"
#include "allocator.h"
#include "ops.h"
#include <numeric>
namespace ainl::core {
/*
in this function, you should define ,size ,type, strides,shape,and allocator the
buffer.because we won't invoke ops.cpp's function.Each Array will be instanced
in Unary<primitive>, and just be construct by prim and inputs.
*/
void addPrimShapeInferContract(const Array &lhs, const Array rhs,
                               Array &output) {

    Dtype lhsDtype = lhs.dtype();
    Dtype rhsDtype = lhs.dtype();
    Dtype output_type = lhs.dtype() < rhs.dtype() ? lhs.dtype() : rhs.dtype();
    std::vector<int> lhsShape = lhs.shape();
    std::vector<int> rhsShape = rhs.shape();
    assert(lhsShape.size() == rhsShape.size());
    auto itemsize = dtypeSize(output_type);
    auto size = std::accumulate(lhsShape.begin(), lhsShape.end(), 1,
                                [](int &a, int &b) { return a * b; }) *
                itemsize;
    output.setDataWithBuffer(allocator::malloc(size), output_type, lhsShape,
                             getStridesFromShape(lhsShape, itemsize));
}
void transposePrimShapeInferContract(const Array &input, Array &output) {

    Dtype lhsDtype = input.dtype();
    Dtype output_type = lhsDtype;
    std::vector<int> inputShape = input.shape();
    std::reverse(inputShape.begin(), inputShape.end());
    auto itemsize = dtypeSize(output_type);
    output.setDataWithBuffer(allocator::malloc(input.size()), output_type,
                             inputShape,
                             getStridesFromShape(inputShape, itemsize));
}

void broadcastPrimShapeInferContract(const Array &input, Array &output,
                                     std::vector<int> shape_) {

    Dtype lhsDtype = input.dtype();
    Dtype output_type = lhsDtype;
    std::vector<int> inputShape = input.shape();
    std::cerr << "+++++++++++++++++++++++";
}

template <typename T, typename... TArgs>
PrimShapeInferContract<T, TArgs...>::PrimShapeInferContract() {

    registerPrimShapeInferContract(
        "add", [](const std::vector<Array> &inputs, Array &output) {
            addPrimShapeInferContract(inputs[0], inputs[1], output);
        });
    registerPrimShapeInferContract(
        "transpose", [](const std::vector<Array> &inputs, Array &output) {
            transposePrimShapeInferContract(inputs[0], output);
        });
    registerPrimShapeInferContract(
        "broadcast", [](const std::vector<Array> &inputs, Array &output,
                        std::vector<int> shape_) {
            broadcastPrimShapeInferContract(inputs[0], output, shape_);
        });
}

PrimShapeInferContract<void, const std::vector<Array> &, Array &> &
getPrimShapeInferContract() {

    static PrimShapeInferContract<void, const std::vector<Array> &, Array &>
        g_primShapeInferContract;
    return g_primShapeInferContract;
}

void resolvePrimShapeInferContract(const std::string &name,
                                   const std::vector<Array> &inputs,
                                   Array &output) {

    getPrimShapeInferContract().resolvePrimShapeInferContract(name, inputs,
                                                              output);
}
void resolvePrimShapeInferContract(const std::string &name,
                                   const std::vector<Array> &inputs,
                                   Array &output, std::vector<int> attr) {

    getPrimShapeInferContract().resolvePrimShapeInferContract(name, inputs,
                                                              output, attr);
}
} // namespace ainl::core