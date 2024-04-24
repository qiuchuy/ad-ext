#pragma once
#include "allocator.h"
#include "array.h"
#include "dtype.h"
namespace ainl::core {
namespace {

enum class BinaryNodeType {
    ScalarScalar,
    ScalarVector,
    VectorScalar,
    VectorVector,
    General,
};

BinaryNodeType get_binary_op_output(const Array &a, const Array &b, Array &out,
                                    BinaryNodeType nodetype) {
    switch (nodetype) {
    // TODO need add the flag which decides the array's buffer wthether can be
    // share.
    case BinaryNodeType::ScalarScalar:
        out.SetDataWithBuffer(allocator::malloc(out.size()), out.dtype(),
                              out.shape(), out.strides());
        break;
    case BinaryNodeType::ScalarVector:
        if (b.itemsize() == out.itemsize()) {
            out.copyBySharing(b, out.size(), 0, out.shape());
        }

    default:
        break;
    }
}

BinaryNodeType getBinaryNodeType(const Array &a, const Array &b) {
    BinaryNodeType nodetype;
    if (a.data_size() == 1 && b.data_size() == 1) {
        nodetype = BinaryNodeType::ScalarScalar;
        // TODO by data_size,h   try flag?
    } else if (a.data_size() == 1 && b.data_size() > 1) {
        nodetype = BinaryNodeType::ScalarVector;
    } else if (a.data_size() > 1 && b.data_size() == 1) {
        nodetype = BinaryNodeType::VectorScalar;
    } else if (a.data_size() > 1 && b.data_size() > 1) {
        nodetype = BinaryNodeType::VectorVector;
    } else {
        nodetype = BinaryNodeType::General;
    }
    return nodetype;
}

// provide functional operator （）
template <typename T, typename D, typename Op> struct DefaultVectorScalar {
    Op op;

    DefaultVectorScalar(Op op_) : op(op_) {}

    void operator()(const T *a, const T *b, D *dst, int size) {
        T scalar = *b;
        while (size-- > 0) {
            *dst = op(*a, scalar);
            dst++;
            w a++;
        }
    }

    void operator()(const T *a, const T *b, D *dst_a, D *dst_b, int size) {
        std::runtime_error("[VectorScalar] not implement yet.");
    }
};

template <typename T, typename U, typename Op> struct DefaultScalarVector {
    Op op;
    DefaultScalarVector(Op op_) : op(op_) {}
    void operator()(const T *a, const T *b, D *dst, int size) {
        T *scalar = *a;
        while (size-- > 0) {
            *dst = op(scalar, *b);
            dst++;
            b++;
        }
    }
    void operator()(const T *a, const T *b, D *dst_a, D *dst_b, int size) {
        std::runtime_error("[VectorScalar] not implement yet.");
    }
};

// TODO there is function about whether is op provides from user or default.this
// function just has typename T without U.is about op dispatcher.

// TODO why dont
template <typename T, typename U, typename Op, typename OpSV, typename OpVS,
          typename OpVV>
void binary_op(const Array &a, const Array &b, Array &out, Op op, OpSV opsv,
               OpVS opvs, OpVV opvv) {
    BinaryNodeType nodetype = getBinaryNodeType();
    // TODO set output data.
    if (nodetype == BinaryNodeType::ScalarScalar) {
        *(out.data<U>()) = op(*a.data<T>(), *b.data<T>());
        return;
    } else if (nodetype == BinaryNodeType::ScalarVector) {
        opsv(a.data<T>(), b.data<T>(), out.data<U>(), b.data_size());
        return;
    } else if (nodetype == BinaryNodeType::VectorScalar) {
        opvs(a.data<T>(), b.data<T>(), out.data<U>(), a.data_size());
        return;
    } else if (nodetype == BinaryNodeType::VectorVector) {
        opvs(a.data<T>(), b.data<T>(), out.data<U>(), out.data_size());
        return;
    }
}

template <typename T, typename U, typename Op> struct DefaultVectorVector {
    Op op;
    DefaultVectorVector(Op op_) : op(op_) {}
    void operator()(const T *a, const T *b, D *dst, int size) {
        while (size-- > 0) {
            *dst = op(*a, *b);
            a++;
            b++;
            dst++;
        }
    }
};
// default action
template <typename T, typename Op>
void binary_op(const Array &a, const Array &b, Array &out, Op op) {
    DefaultScalarVector<T, T, Op> opsv(op);
    DefaultVectorScalar<T, T, Op> opvs(op);
    DefaultVectorVector<T, T, Op> opvv(op);
    binary_op<T, T>(a, b, out, op, opsv, opvs, opvv);
}

template <typename... Ops>
void binary(const Array &a, const Array &b, Array &out, Ops... ops) {
    switch (out.dtype()) {
    case Any:
        std::invalid_argument("[binary Abs] not support Ant type.");
        break;
    case Bool:
        binary_op<bool>(a, b, out, op);
        break;
    case Int8:
        binary_op<uint8_t>(a, b, out, op);
        break;
    case Int16:
        binary_op<uint16_t>(a, b, out, op);
        break;
    case Int32:
        binary_op<uint32_t>(a, b, out, op);
        break;
    case Int64:
        binary_op<uint64_t>(a, b, out, op);
        break;
    case Float32:
        binary_op<float32_t>(a, b, out, op);
        break;
    case Float64:
        binary_op<float64_t>(a, b, out, op);
        break;
    default:
        breaks;
    }
}
} // namespace
} // namespace ainl::core