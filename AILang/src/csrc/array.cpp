#include "array.h"
#include "graph.h"
#include "ops.h"
#include "trace.h"
#include "utils/logger.h"

namespace ainl::core {

Array::Array(const allocator::Buffer &buffer, Dtype dtype,
             std::function<void(allocator::Buffer)> deleter)
    : data_(std::make_shared<Data>(buffer, deleter)), dtype_(dtype) {}

template <typename T> Array::Array(T val, Dtype dtype) : dtype_(dtype) {
    auto buffer = allocator::malloc(sizeof(T));
    *static_cast<T *>(buffer.ptr()) = val;
    data_ = std::make_shared<Data>(
        buffer, [](allocator::Buffer buffer) { allocator::free(buffer); });
}

template <typename T>
Array::Array(std::initializer_list<T> list, Dtype dtype) : dtype_(dtype) {
    auto buffer = allocator::malloc(sizeof(T) * list.size());
    auto ptr = static_cast<T *>(buffer.ptr());
    for (auto &val : list) {
        *ptr = val;
        ptr++;
    }
    data_ = std::make_shared<Data>(
        buffer, [](allocator::Buffer buffer) { allocator::free(buffer); });
}

Array::Array(Dtype dtype, std::shared_ptr<Primitive> prim,
             std::vector<Array> inputs)
    : info_(std::make_shared<MetaData>(prim, inputs)), dtype_(dtype) {}

void Array::eval() {
    auto trace = getTopTrace();
    std::function<void(Array &)> recursion = [&](Array &arr) -> void {
        if (evaluated()) {
            return;
        } else {
            for (auto &input : arr.inputs()) {
                recursion(input);
            }
            trace->process(arr.primitive(), arr.inputs(), arr);
        }
    };
    recursion(*this);
}

Array::ArrayIterator::ArrayIterator(const Array &arr, int idx)
    : arr(arr), idx(idx) {
    // if (arr.ndim() == 0) {
    //     throw std::invalid_argument("Cannot iterate over 0-d array.");
    // }
}

Array::ArrayIterator::reference Array::ArrayIterator::operator*() const {
    auto shape = arr.shape();
    shape.erase(shape.begin());
    int start = idx;
    int end = idx + 1;
    return reshape(slice(arr, start, end, 1), shape);
};

// ConcreteArray::ConcreteArray(const Array &tracer) : Array(tracer) {
//     tracer_ = std::make_shared<Array>(tracer);
// }

Array makeArrayFromScalar(int val) { return Array(val); }

} // namespace ainl::core
