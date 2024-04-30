#include <map>
#include <sstream>

#include "array.h"
#include "graph.h"
#include "ops.h"
#include "primitive.h"
#include "trace.h"
#include "utils/logger.h"

namespace ainl::core {

Array::Array(Dtype dtype, std::shared_ptr<Primitive> prim,
             const std::vector<Array> &inputs, const std::vector<int> &shape,
             const std::vector<int> &stride)
    : dtype_(dtype) {
    std::vector<std::shared_ptr<Tracer>> inputTracers;
    for (const auto &input : inputs) {
        inputTracers.push_back(
            std::dynamic_pointer_cast<Tracer>(std::make_shared<Array>(input)));
    }
    inputs_ = inputTracers;
    prim_ = std::move(prim);
    shape_ = std::make_shared<std::vector<int>>(shape);
    stride_ = std::make_shared<std::vector<int>>(stride);
    // add attribute size_
    size_ =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
        dtypeSize(dtype);
}

Array::Array(const std::vector<std::shared_ptr<Tracer>> &inputs,
             const std::shared_ptr<Primitive> &prim)
    : Tracer(inputs, prim) {}

Array::Array(const allocator::Buffer &buffer, Dtype dtype,
             const std::vector<int> &shape, const std::vector<int> &stride)
    : Tracer({}, std::make_shared<IdentityPrimitive>()),
      data_(std::make_shared<Data>(
          buffer, [](allocator::Buffer buffer) { allocator::free(buffer); })),
      dtype_(dtype), shape_(std::make_shared<std::vector<int>>(shape)),
      stride_(std::make_shared<std::vector<int>>(stride)) {
    ptr_ = buffer.ptr();
    size_ =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
        dtypeSize(dtype);
}

void Tracer::eval() {
    // LOG_DEBUG("%s", "[eval] Start evaluating tracer");
    auto trace = getCurrentTrace();
    std::function<void(std::shared_ptr<Tracer> tracer)> recursion =
        [&](std::shared_ptr<Tracer> tracer) -> void {
        if (evaluated()) {
            return;
        } else {
            for (auto &input : tracer->inputs()) {
                recursion(input);
            }
            if (!tracer->isLeaf()) {
                // LOG_DEBUG("[eval] Evaluating tracer with primitive %s",
                //   tracer->primitive()->toString().c_str());
                trace->process(tracer->primitive(), tracer->inputs(), tracer);
            }
        }
    };

    // LOG_DEBUG("%s", std::string("[eval] Current program transformation: " +
    //                             trace->toString())
    //                     .c_str());
    recursion(shared_from_this());
}

bool Tracer::evaluated() const { return false; }

std::string Tracer::toString() const { return "tracer"; }

std::vector<std::shared_ptr<Tracer>> Tracer::subtracers() const { return {}; }

// CopyBySharing may cause the old Array coverd by new Array
void Array::copyBySharing(const Array &other, size_t size, size_t offset,
                          const std::vector<int> &shape) {
    data_ = other.data_;
    ptr_ = (char *)other.ptr_ + offset;
    shape_ = std::make_shared<std::vector<int>>(shape);
    dtype_ = other.dtype_;
    size_ = size;
    inputs_ = other.inputs_;
    prim_ = other.prim_;
    auto stride = std::vector<int>(shape.size(), 1);
    for (size_t i = 0; i < shape.size(); i++) {
        if (i == shape.size() - 1) {
            stride[i] = dtypeSize(dtype_);
        } else {
            for (size_t j = i + 1; j < shape.size(); j++) {
                stride[i] *= shape[j] * dtypeSize(dtype_);
            }
        }
    }
    stride_ = std::make_shared<std::vector<int>>(stride);
}

// TODO need to be fix
void Array::SetDataWithBuffer(allocator::Buffer buffer, Dtype dtype,
                              const std::vector<int> &shape,
                              const std::vector<int> &stride) {
    data_ = std::make_shared<Data>(
        buffer, [](allocator::Buffer buffer) { allocator::free(buffer); });
    ptr_ = buffer.ptr();
    dtype_ = dtype;
    shape_ = std::make_shared<std::vector<int>>(shape);
    size_ =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
        dtypeSize(dtype);
    stride_ = std::make_shared<std::vector<int>>(stride);
}

Array::ArrayIterator::ArrayIterator(const Array &arr, int idx)
    : arr(arr), idx(idx) {
    if (arr.ndim() == 0) {
        throw std::invalid_argument("Cannot iterate over 0-d array.");
    }
}

Array::ArrayIterator::reference Array::ArrayIterator::operator*() const {
    auto shape = arr.shape();
    auto stride = std::vector<int>(arr.shape().size(), 1);
    shape.erase(shape.begin());
    auto start = std::vector<int>(arr.shape().size(), 0);
    start[0] = idx;
    auto end = arr.shape();
    end[0] = idx + 1;
    return reshape(slice(arr, start, end, stride), shape);
};

std::ostream &operator<<(std::ostream &os, const Array &arr) {
    os << "Array(";
    switch (arr.dtype().type) {
    case Dtype::DataType::BoolType:
        arr.print<bool>(os, 0, 0);
        break;
    case Dtype::DataType::Int8Type:
        arr.print<int8_t>(os, 0, 0);
        break;
    case Dtype::DataType::Int16Type:
        arr.print<int16_t>(os, 0, 0);
        break;
    case Dtype::DataType::Int32Type:
        arr.print<int32_t>(os, 0, 0);
        break;
    case Dtype::DataType::Int64Type:
        arr.print<int64_t>(os, 0, 0);
        break;
    case Dtype::DataType::Float32Type:
        arr.print<float>(os, 0, 0);
        break;
    case Dtype::DataType::Float64Type:
        arr.print<double>(os, 0, 0);
        break;
    }
    os << ")";
    return os;
}

std::string Array::toString() const {
    std::ostringstream oss;
    oss << *this;
}

} // namespace ainl::core
