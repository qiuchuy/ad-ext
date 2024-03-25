#include "array.h"
#include "graph.h"
#include "ops.h"
#include "trace.h"
#include "utils/logger.h"

namespace ainl::core {

Array::Array(Dtype dtype, std::shared_ptr<Primitive> prim,
             std::vector<Array> inputs)
    : info_(std::make_shared<MetaData>(prim, inputs)), dtype_(dtype) {}

Array::Array(const allocator::Buffer &buffer, Dtype dtype,
             const std::vector<int> &shape, const std::vector<int> &stride)
    : data_(std::make_shared<Data>(
          buffer, [](allocator::Buffer buffer) { allocator::free(buffer); })),
      dtype_(dtype), shape_(std::make_shared<std::vector<int>>(shape)),
      stride_(std::make_shared<std::vector<int>>(stride)) {
  size_ =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
      dtypeSize(dtype);
}

void Array::eval() {
  DEBUG("[eval] Start evaluating array.")
  while (hasRemainingTrace()) {
    auto trace = popLastTrace();
    std::function<void(Array &)> recursion = [&](Array &arr) -> void {
      if (evaluated()) {
        return;
      } else {
        for (auto &input : arr.inputs()) {
          recursion(input);
        }
        DEBUG("[eval] Processing array with primitive" +
              arr.primitive()->toString())
        trace->process(arr.primitive(), arr.inputs(), arr);
      }
    };
    recursion(*this);
  }
}

void Array::copyBySharing(const Array &other, size_t size, size_t offset,
                          const std::vector<int> &shape) {
  data_->ptr() = other.data_->ptr() + offset;
  shape_ = std::make_shared<std::vector<int>>(shape);
  size_ = size;
  info_ = other.info_;
  dtype_ = other.dtype_;
}

Array::ArrayIterator::ArrayIterator(const Array &arr, int idx)
    : arr(arr), idx(idx) {
  if (arr.ndim() == 0) {
    throw std::invalid_argument("Cannot iterate over 0-d array.");
  }
}

Array::ArrayIterator::reference Array::ArrayIterator::operator*() const {
  auto shape = arr.shape();
  shape.erase(shape.begin());
  auto stride = std::vector<int>(shape.size(), 1);
  auto start = arr.shape();
  start[0] = idx;
  auto end = arr.shape();
  end[0] = idx + 1;
  return reshape(slice(arr, start, end, stride), shape);
};

std::ostream &operator<<(std::ostream &os, Array &arr) {
  if (!arr.evaluated()) {
    arr.eval();
  }
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
  return os;
}

// ConcreteArray::ConcreteArray(const Array &tracer) : Array(tracer) {
//     tracer_ = std::make_shared<Array>(tracer);
// }

} // namespace ainl::core
