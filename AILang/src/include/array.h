#pragma once

#include <functional>
#include <memory>
#include <numeric>
#include <variant>
#include <vector>

#include "allocator.h"
#include "device.h"
#include "dtype.h"
#include "graph.h"
#include "primitive.h"
#include "trace.h"

namespace ainl::core {

class Array;
class Primitive;

class MetaData {
public:
  MetaData() = default;
  MetaData(std::shared_ptr<Primitive> prim, std::vector<Array> inputs)
      : prim_(std::move(prim)), inputs_(std::move(inputs)) {}
  virtual ~MetaData() = default;
  friend class Array;

private:
  std::shared_ptr<Primitive> prim_;
  std::vector<Array> inputs_;
  std::shared_ptr<Array> tracer_;
};

class Array {
public:
  /* Construct a scalar array*/
  template <typename T> explicit Array(T val, Dtype dtype = TypeToDtype<T>()) {
    auto buffer = allocator::malloc(sizeof(T));
    ptr_ = buffer.ptr();
    data_ = std::make_shared<Data>(
        buffer, [](allocator::Buffer buffer) { allocator::free(buffer); });
    dtype_ = dtype;
    size_ = sizeof(T);
    shape_ = std::make_shared<std::vector<int>>();
    info_ = std::make_shared<MetaData>();
    *(reinterpret_cast<T *>(ptr_)) = val;
    stride_ = std::make_shared<std::vector<int>>();
    LOG_DEBUG("[malloc] Fill address %d with value: %d",
              reinterpret_cast<uintptr_t>(ptr_), val);
  }

  template <typename T>
  /* Construct an array from a flattened vector*/
  Array(const std::vector<T> &vec, const std::vector<int> &shape,
        Dtype dtype = TypeToDtype<T>())
      : shape_(std::make_shared<std::vector<int>>(shape)) {
    auto buffer = allocator::malloc(sizeof(T) * vec.size());
    ptr_ = buffer.ptr();
    data_ = std::make_shared<Data>(
        buffer, [](allocator::Buffer buffer) { allocator::free(buffer); });
    dtype_ = dtype;
    size_ = vec.size() * sizeof(T);
    info_ = std::make_shared<MetaData>();
    stride_ = std::make_shared<std::vector<int>>(shape.size(), 1);
    std::copy(vec.begin(), vec.end(), reinterpret_cast<T *>(ptr_));
  }

  /* Construct an array from buffer*/
  Array(const allocator::Buffer &buffer, Dtype dtype,
        const std::vector<int> &shape, const std::vector<int> &stride);
  /* Construct an array by copy*/
  Array(const Array &other) = default;

  /* Construct an array in the computational graph*/
  Array(Dtype dtype, std::shared_ptr<Primitive> prim, std::vector<Array> inputs,
        const std::vector<int> &shape, const std::vector<int> &stride);

  struct Data {
    allocator::Buffer buffer;
    std::function<void(allocator::Buffer)> deleter;
    Data(const allocator::Buffer &buffer,
         std::function<void(allocator::Buffer)> deleter)
        : buffer(buffer), deleter(deleter) {}
    Data(const Data &data) = delete;
    Data &operator=(const Data &data) = delete;
    ~Data() { deleter(buffer); }
  };

  void eval();

  bool evaluated() const { return data_ != nullptr; }

  bool isLeaf() const { return info_->inputs_.empty(); }

  void copyBySharing(const Array &array, size_t size, size_t offset,
                     const std::vector<int> &shape);

  struct ArrayIterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = const Array;
    using reference = value_type;

    explicit ArrayIterator(const Array &arr, int idx);

    reference operator*() const;

    ArrayIterator &operator+(difference_type diff) {
      idx += diff;
      return *this;
    }

    ArrayIterator &operator++() {
      idx++;
      return *this;
    }

    ArrayIterator operator++(int) {
      ArrayIterator temp = *this;
      idx++;
      return temp;
    }

    friend bool operator==(const ArrayIterator &a, const ArrayIterator &b) {
      return a.idx == b.idx;
    };
    friend bool operator!=(const ArrayIterator &a, const ArrayIterator &b) {
      return !(a == b);
    };

  private:
    const Array &arr;
    int idx;
  };

  ArrayIterator begin() const { return ArrayIterator(*this, 0); }
  ArrayIterator end() const {
    return ArrayIterator(*this, this->shape_->at(0));
  }

  std::shared_ptr<Array> tracer() { return info_->tracer_; }
  std::shared_ptr<Primitive> primitive() const { return info_->prim_; }
  std::vector<Array> &inputs() { return info_->inputs_; }
  std::vector<int> shape() const { return *(shape_); }
  std::vector<int> strides() const { return *(stride_); }
  size_t size() const { return size_; }
  size_t itemsize() const { return dtypeSize(dtype_); }
  Dtype dtype() const { return dtype_; }
  size_t ndim() const { return shape_->size(); }

  template <typename T> T *data() { return static_cast<T *>(ptr_); };

  template <typename T> const T *data() const {
    return static_cast<T *>(ptr_);
  };

  friend std::ostream &operator<<(std::ostream &os, const Array &arr);

  template <typename T>
  void print(std::ostream &os, size_t offset, size_t dim) const {
    if (ndim() == 0) {
      os << *reinterpret_cast<uintptr_t *>((char*)ptr_ + offset / itemsize());
      return;
    }
    os << "[";
    if (dim == ndim() - 1) {
      LOG_DEBUG("[print] Printing array at %d with offset %d",
                reinterpret_cast<uintptr_t>(ptr_), offset);
      for (size_t i = 0; i < shape_->at(dim); i++) {
        os << (*(data<T>() + offset / itemsize() + i));
        if (i != shape_->at(dim) - 1) {
          os << ", ";
        }
      }

    } else {
      for (size_t i = 0; i < shape_->at(dim); i++) {
        auto dimOffset =
            std::accumulate(shape_->begin() + dim + 1, shape_->end(), 1,
                            std::multiplies<int>());
        print<T>(os, offset + i * dimOffset * itemsize(), dim + 1);
        if (i != shape_->at(dim) - 1) {
          os << "\n";
        }
      }
    }
    os << "]";
  }

protected:
  std::shared_ptr<Data> data_;
  Dtype dtype_;
  size_t size_;
  std::shared_ptr<MetaData> info_;
  std::shared_ptr<std::vector<int>> shape_;
  std::shared_ptr<std::vector<int>> stride_;
  void *ptr_;
}; // namespace ainl::core

/*
class ConcreteArray : public Array {
  public:
    template <typename T>
    explicit ConcreteArray(T val, Dtype dtype = TypeToDtype<T>())
        : Array(val, dtype) {}

    ConcreteArray(const allocator::Buffer &buffer, Dtype dtype,
                  std::function<void(allocator::Buffer)> deleter)
        : Array(buffer, dtype, deleter) {}

    ConcreteArray(const Array &array);

    ConcreteArray(const std::vector<int> &shape, const std::vector<int> &stride,
                  int offset, int ndim, std::shared_ptr<Primitive> prim,
                  std::vector<Array> inputs);
};

class ConcreteArrayMetaData : public MetaData {
  public:
    ConcreteArrayMetaData() = default;

    std::vector<size_t> shape() const { return shape_; }
    std::vector<size_t> stride() const { return stride_; }
    size_t offset() const { return offset_; }
    size_t ndim() const { return ndim_; }

    MetaData::TraceMode getTraceMode() override {
        return MetaData::TraceMode::eval;
    }

  private:
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    size_t offset_;
    size_t ndim_;
};
*/

} // namespace ainl::core
