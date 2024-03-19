#include <numeric>

#include "array.h"
#include "graph.h"
#include "ops.h"
#include "trace.h"
#include "utils/logger.h"

namespace ainl::core {

Array::Array(const allocator::Buffer &buffer, Dtype dtype,
             std::function<void(allocator::Buffer)> deleter)
    : data_(std::make_shared<Data>(buffer, deleter)), dtype_(dtype) {}

Array::Array(Dtype dtype, std::shared_ptr<Primitive> prim,
             std::vector<Array> inputs)
    : info_(std::make_shared<MetaData>(prim, inputs)), dtype_(dtype) {}

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

void Array::print(std::ostream &os, size_t offset, size_t dim) {
  DEBUG("[print] Printing array at " +
        std::to_string(reinterpret_cast<uintptr_t>(data_->ptr())) +
        " with offset: " + std::to_string(offset))
  os << "Array[";
  if (ndim() == 0) {
    os << *static_cast<size_t *>(data_->ptr());
    return;
  }

  if (dim == ndim() - 1) {
    os << *static_cast<size_t *>(data_->ptr() + offset);
  } else {
    os << "[";
    for (size_t i = 0; i < shape_->at(dim); i++) {
      auto dimOffset = std::accumulate(shape_->begin() + dim + 1, shape_->end(),
                                       1, std::multiplies<int>());
      print(os, offset + i * dimOffset, dim + 1);
      if (i != shape_->at(dim) - 1) {
        os << ", ";
      }
    }
    os << "]";
  }
  os << "]";
}

std::ostream &operator<<(std::ostream &os, Array &arr) {
  if (!arr.evaluated()) {
    arr.eval();
  }
  arr.print(os, 0, 0);
  return os;
}

// ConcreteArray::ConcreteArray(const Array &tracer) : Array(tracer) {
//     tracer_ = std::make_shared<Array>(tracer);
// }

} // namespace ainl::core
