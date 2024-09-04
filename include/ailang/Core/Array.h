#pragma once

#include <functional>
#include <memory>
#include <numeric>
#include <sys/types.h>
#include <variant>
#include <vector>

#include "ailang/Core/Allocator.h"
#include "ailang/Core/Device.h"
#include "ailang/Core/Dtype.h"
#include "ailang/Core/Graph.h"
#include "ailang/Core/Trace.h"
#include "ailang/IR/Type.h"
#include "ailang/Utils/Logger.h"

namespace ainl::core {

class Array;
class Primitive;
class BaseTrace;

class Tracer : public std::enable_shared_from_this<Tracer> {
public:
  enum class TracerTy {
    ArrayTy = 0,
    JVPTracerTy,
    JITTracerTy,
  };
  Tracer() = default;
  virtual ~Tracer() = default;
  Tracer(const std::vector<std::shared_ptr<Tracer>> &inputs,
         const std::shared_ptr<Primitive> &prim);
  Tracer(const std::vector<std::shared_ptr<Tracer>> &inputs,
         const std::shared_ptr<Primitive> &prim, uint64_t idx);
  bool isLeaf() { return inputs_.empty(); }
  std::vector<std::shared_ptr<Tracer>> inputs() { return inputs_; }
  std::shared_ptr<Primitive> primitive() const { return prim_; }
  void eval();
  virtual TracerTy getTracerTy() const = 0;
  virtual ir::TypePtr getJITType() = 0;
  virtual std::shared_ptr<Tracer> aval() = 0;
  virtual bool evaluated() const = 0;
  virtual std::string toString() const = 0;
  virtual std::shared_ptr<Tracer> clone() = 0;
  std::vector<std::shared_ptr<Tracer>> outputs() {
    std::vector<std::shared_ptr<Tracer>> outputs;
    outputs.reserve(siblings_.size() + 1);
    outputs.insert(outputs.end(), siblings_.begin(), siblings_.begin() + idx_);
    outputs.push_back(shared_from_this());
    outputs.insert(outputs.end(), siblings_.begin() + idx_, siblings_.end());
    return outputs;
  }
  std::shared_ptr<BaseTrace> trace() { return trace_; }
  void setSiblings(const std::vector<std::shared_ptr<Tracer>> &siblings) {
    siblings_ = siblings;
  }
  void setIdx(uint64_t idx) { idx_ = idx; }
  bool operator==(Tracer &other);
  bool operator>(Tracer &other);

  template <typename T> bool operator==(T scalar) {
    if (auto array = std::dynamic_pointer_cast<Array>(shared_from_this())) {
      return *array == scalar;
    }
    return aval()->operator==(scalar);
  }

  template <typename T> bool operator>(T scalar) {
    if (auto array = std::dynamic_pointer_cast<Array>(shared_from_this())) {
      return *array > scalar;
    }
    return aval()->operator>(scalar);
  }

  operator std::string() { return toString(); }

protected:
  std::vector<std::shared_ptr<Tracer>> inputs_;
  std::vector<std::shared_ptr<Tracer>> siblings_;
  std::shared_ptr<Primitive> prim_;
  std::shared_ptr<BaseTrace> trace_;
  uint64_t idx_ = 0;

private:
};

class Array : public Tracer {
public:
  /* Construct a scalar array*/
  template <typename T>
  explicit Array(T val, Dtype dtype = TypeToDtype<T>(), Device device = cpu)
      : Tracer({}, nullptr) {
    LOG_DEBUG("initialized value: %f", val);
    auto buffer = allocator::malloc(sizeof(T));
    ptr_ = buffer.ptr();
    data_ = std::make_shared<Data>(
        buffer, [](allocator::Buffer buffer) { allocator::free(buffer); });
    device_ = device;
    dtype_ = dtype;
    size_ = sizeof(T);
    shape_ = std::make_shared<std::vector<int>>();
    *(reinterpret_cast<T *>(ptr_)) = val;
    stride_ = std::make_shared<std::vector<int>>();
    trace_ = getStandardEvalTrace();
    LOG_DEBUG("[malloc] Fill address %d with value: %f",
              reinterpret_cast<uintptr_t>(ptr_), val);
  }

  template <typename T>
  /* Construct an array from a flattened vector*/
  Array(const std::vector<T> &vec, const std::vector<int> &shape, Device device,
        Dtype dtype = TypeToDtype<T>())
      : Tracer({}, nullptr), shape_(std::make_shared<std::vector<int>>(shape)) {
    auto buffer = allocator::malloc(sizeof(T) * vec.size());
    ptr_ = buffer.ptr();
    data_ = std::make_shared<Data>(
        buffer, [](allocator::Buffer buffer) { allocator::free(buffer); });
    device_ = cpu;
    dtype_ = dtype;
    size_ = vec.size() * sizeof(T);
    trace_ = getStandardEvalTrace();
    for (size_t i = 0; i < vec.size(); ++i) {
      *(data<T>() + i) = vec[i];
    }
    std::vector<int> strides;
    for (size_t i = 0; i < shape.size(); i++) {
      int stride = 1;
      for (size_t j = i + 1; j < shape.size(); j++) {
        stride *= shape[j];
      }
      strides.push_back(stride * sizeof(T));
    }
    stride_ = std::make_shared<std::vector<int>>(strides);
  }

  /* Construct an array from buffer*/
  Array(const allocator::Buffer &buffer, Dtype dtype,
        const std::vector<int> &shape, const std::vector<int> &stride,
        Device device = cpu);
  /* Construct an array by copy*/
  Array(const Array &other) = default;

  /* Construct an array in the computational graph in eager mode*/
  Array(Dtype dtype, std::shared_ptr<Primitive> prim,
        const std::vector<Array> &inputs, const std::vector<int> &shape,
        const std::vector<int> &stride);

  /* Construct an array in the computational graph in the middle of higher order
   * program transformation*/
  /* Data members will be initialized in a lazy style during the evaluation
   * procedure of program transformation*/
  Array(const std::vector<std::shared_ptr<Tracer>> &inputs,
        const std::shared_ptr<Primitive> &prim);

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

  bool evaluated() const override { return data_ != nullptr; }

  Tracer::TracerTy getTracerTy() const override { return TracerTy::ArrayTy; }

  std::shared_ptr<Tracer> aval() override { return shared_from_this(); }

  void copyBySharing(const Array &array, size_t size, size_t offset,
                     const std::vector<int> &shape,
                     const std::vector<int> &stride = {});
  void setDataWithBuffer(allocator::Buffer buffer, Dtype dtype,
                         const std::vector<int> &shape,
                         const std::vector<int> &stride);

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

  std::shared_ptr<Primitive> primitive() const { return prim_; }
  std::vector<int> shape() const { return *(shape_); }
  std::vector<int> strides() const { return *(stride_); }
  size_t size() const { return size_; }
  size_t itemsize() const { return dtypeSize(dtype_); }
  Dtype dtype() const { return dtype_; }
  size_t ndim() const { return shape_->size(); }
  Device device() const { return device_; }

  template <typename T> T item() {
    if (!evaluated()) {
      eval();
    }
    if (ndim() != 0) {
      throw std::runtime_error("Item() is only supported for scalar array.");
    }
    return *(data<T>());
  }

  template <typename T> T *data() { return static_cast<T *>(ptr_); };

  template <typename T> const T *data() const {
    return static_cast<T *>(ptr_);
  };

  ir::TypePtr getJITType() override;

  std::shared_ptr<Tracer> clone() override {
    return std::make_shared<Array>(*this);
  }

  friend std::ostream &operator<<(std::ostream &os, const Array &arr);

  std::string toString() const override;

  template <typename T>
  void print(std::ostream &os, size_t offset, size_t dim) const {
    if (ndim() == 0) {
      os << (*(data<T>() + offset / itemsize()));
      return;
    }
    os << "[";
    if (dim == ndim() - 1) {
      LOG_DEBUG("[print] Printing array at %d with offset %d",
                reinterpret_cast<uintptr_t>(ptr_), offset);
      for (size_t i = 0; i < shape_->at(dim); i++) {
        os << (*(data<T>() + (offset + i * stride_->at(dim)) / itemsize()));
        if (i != shape_->at(dim) - 1) {
          os << ", ";
        }
      }

    } else {
      for (size_t i = 0; i < shape_->at(dim); i++) {
        print<T>(os, offset + i * stride_->at(dim), dim + 1);
        if (i != shape_->at(dim) - 1) {
          os << ",";
        }
      }
    }
    os << "]";
  }

  bool operator==(Array &other) {
    if (!evaluated())
      eval();
    if (!other.evaluated())
      other.eval();

    bool shapeEqual =
        std::equal(shape_->begin(), shape_->end(), other.shape_->begin());
    bool strideEqual =
        std::equal(stride_->begin(), stride_->end(), other.stride_->begin());
    if (!shapeEqual || !strideEqual || dtype_.type != other.dtype_.type ||
        size_ != other.size_) {
      return false;
    }

    if (ndim() == 0) {
      throw std::runtime_error(
          "Comparison is only supported for scalar array.");
    }

    switch (dtype_.type) {
    case Dtype::DataType::BoolType:
      return item<bool>() == other.item<bool>();
    case Dtype::DataType::Int16Type:
      return item<int16_t>() == other.item<int16_t>();
    case Dtype::DataType::Int32Type:
      return item<int32_t>() == other.item<int32_t>();
    case Dtype::DataType::Int64Type:
      return item<int64_t>() == other.item<int64_t>();
    case Dtype::DataType::Float32Type:
      return item<float>() == other.item<float>();
    case Dtype::DataType::Float64Type:
      return item<double>() == other.item<double>();
    default:
      throw std::runtime_error("Unsupported data type.");
    }
  }

  bool operator>(Array &other) {
    if (!evaluated())
      eval();
    if (!other.evaluated())
      other.eval();

    if (ndim() != 0) {
      throw std::runtime_error(
          "Comparison is only supported for scalar array.");
    }

    switch (dtype_.type) {
    case Dtype::DataType::BoolType:
      return item<bool>() > other.item<bool>();
    case Dtype::DataType::Int16Type:
      return item<int16_t>() > other.item<int16_t>();
    case Dtype::DataType::Int32Type:
      return item<int32_t>() > other.item<int32_t>();
    case Dtype::DataType::Int64Type:
      return item<int64_t>() > other.item<int64_t>();
    case Dtype::DataType::Float32Type:
      return item<float>() > other.item<float>();
    case Dtype::DataType::Float64Type:
      return item<double>() > other.item<double>();
    default:
      throw std::runtime_error("Unsupported data type.");
    }
  }

  template <typename T> bool operator==(T scalar) {
    return item<T>() == scalar;
  }

  template <typename T> bool operator>(T scalar) { return item<T>() > scalar; }

protected:
  std::shared_ptr<Data> data_;
  Dtype dtype_;
  Device device_;
  size_t size_;
  std::shared_ptr<std::vector<int>> shape_;
  std::shared_ptr<std::vector<int>> stride_;
  void *ptr_;
};

template <typename T>
std::vector<T>
convertTracerVector(const std::vector<std::shared_ptr<Tracer>> &sharedPtrVec) {
  std::vector<T> result;
  result.reserve(sharedPtrVec.size());

  for (auto &ptr : sharedPtrVec) {
    result.push_back(*std::dynamic_pointer_cast<T>(ptr));
  }
  return result;
}

template <typename T>
std::vector<std::shared_ptr<Tracer>>
convertTracerSharedPtrVector(const std::vector<T> &inputs) {
  std::vector<std::shared_ptr<Tracer>> result;
  for (auto &input : inputs) {
    result.push_back(std::make_shared<T>(input));
  }
  return result;
}

template <typename T>
std::shared_ptr<T> asTracer(const std::shared_ptr<Tracer> &tracer) {
  if (auto array = std::dynamic_pointer_cast<T>(tracer)) {
    return array;
  } else {
    return nullptr;
  }
}

template <typename T>
std::shared_ptr<T> asTrace(const std::shared_ptr<BaseTrace> &tracer) {
  if (auto array = std::dynamic_pointer_cast<T>(tracer)) {
    return array;
  } else {
    return nullptr;
  }
}

} // namespace ainl::core