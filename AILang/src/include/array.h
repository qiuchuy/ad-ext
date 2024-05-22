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
#include "ir/type.h"
#include "primitive.h"
#include "trace.h"
#include "utils/logger.h"

namespace ainl::core {

class Array;
class Primitive;
class BaseTrace;

class Tracer : public std::enable_shared_from_this<Tracer> {
  public:
    Tracer() = default;
    virtual ~Tracer() = default;
    Tracer(const std::vector<std::shared_ptr<Tracer>> &inputs,
           const std::shared_ptr<Primitive> &prim);
    bool isLeaf() { return inputs_.empty(); }
    std::vector<std::shared_ptr<Tracer>> inputs() { return inputs_; }
    std::shared_ptr<Primitive> primitive() const { return prim_; }
    void eval();
    virtual ir::TypePtr getJITType() = 0;
    virtual std::shared_ptr<Tracer> aval() = 0;
    virtual bool evaluated() const = 0;
    virtual std::string toString() const = 0;

  protected:
    std::vector<std::shared_ptr<Tracer>> inputs_;
    std::shared_ptr<Primitive> prim_;
    std::shared_ptr<BaseTrace> trace_;
};

class Array : public Tracer {
  public:
    /* Construct a scalar array*/
    template <typename T>
    explicit Array(T val, Dtype dtype = TypeToDtype<T>())
        : Tracer({}, nullptr) {
        LOG_DEBUG("initialized value: %f", val);
        auto buffer = allocator::malloc(sizeof(T));
        ptr_ = buffer.ptr();
        data_ = std::make_shared<Data>(
            buffer, [](allocator::Buffer buffer) { allocator::free(buffer); });
        dtype_ = dtype;
        size_ = sizeof(T);
        shape_ = std::make_shared<std::vector<int>>();
        *(reinterpret_cast<T *>(ptr_)) = val;
        stride_ = std::make_shared<std::vector<int>>();
        LOG_DEBUG("[malloc] Fill address %d with value: %f",
                  reinterpret_cast<uintptr_t>(ptr_), val);
    }

    template <typename T>
    /* Construct an array from a flattened vector*/
    Array(const std::vector<T> &vec, const std::vector<int> &shape,
          Dtype dtype = TypeToDtype<T>())
        : Tracer({}, nullptr),
          shape_(std::make_shared<std::vector<int>>(shape)) {
        auto buffer = allocator::malloc(sizeof(T) * vec.size());
        ptr_ = buffer.ptr();
        data_ = std::make_shared<Data>(
            buffer, [](allocator::Buffer buffer) { allocator::free(buffer); });
        dtype_ = dtype;
        size_ = vec.size() * sizeof(T);
        stride_ = std::make_shared<std::vector<int>>(shape.size(), 1);
        std::copy(vec.begin(), vec.end(), reinterpret_cast<T *>(ptr_));
    }

    /* Construct an array from buffer*/
    Array(const allocator::Buffer &buffer, Dtype dtype,
          const std::vector<int> &shape, const std::vector<int> &stride);
    /* Construct an array by copy*/
    Array(const Array &other) = default;

    /* Construct an array in the computational graph in eager mode*/
    Array(Dtype dtype, std::shared_ptr<Primitive> prim,
          const std::vector<Array> &inputs, const std::vector<int> &shape,
          const std::vector<int> &stride);

    /* Construct an array in the computational graph in the middle of higher
     * order program transformation*/
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

    std::shared_ptr<Tracer> aval() override { return shared_from_this(); }

    void copyBySharing(const Array &array, size_t size, size_t offset,
                       const std::vector<int> &shape,
                       const std::vector<int> &stride = {});
    void SetDataWithBuffer(allocator::Buffer buffer, Dtype dtype,
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
    size_t data_size() const { return size_ / itemsize(); }
    size_t itemsize() const { return dtypeSize(dtype_); }
    Dtype dtype() const { return dtype_; }
    size_t ndim() const { return shape_->size(); }

    template <typename T> T *data() { return static_cast<T *>(ptr_); };

    template <typename T> const T *data() const {
        return static_cast<T *>(ptr_);
    };

    ir::TypePtr getJITType() override;

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
                os << (*(data<T>() +
                         (offset + i * stride_->at(dim)) / itemsize()));
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

  protected:
    std::shared_ptr<Data> data_;
    Dtype dtype_;
    size_t size_;
    std::shared_ptr<std::vector<int>> shape_;
    std::shared_ptr<std::vector<int>> stride_;
    void *ptr_;
};

template <typename T1, typename T2>
std::vector<T1>
tracerVectorConversion(const std::vector<std::shared_ptr<T2>> &tracers) {
    std::vector<T1> arrays;
    for (auto &tracer : tracers) {
        if (auto array = std::dynamic_pointer_cast<T1>(tracer)) {
            arrays.push_back(*(std::dynamic_pointer_cast<T1>(tracer)));
        } else {
            throw std::runtime_error("Cannot convert one tracer to another.");
        }
    }
    return arrays;
}

} // namespace ainl::core