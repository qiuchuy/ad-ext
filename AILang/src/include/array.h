#pragma once

#include <functional>
#include <memory>
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
    template <typename T> explicit Array(T val, Dtype dtype = TypeToDtype<T>());

    /* Construct an array from python list/ tuple*/
    template <typename T>
    Array(std::initializer_list<T> list, Dtype dtype = TypeToDtype<T>());

    /* Construct an array from buffer*/
    Array(const allocator::Buffer &buffer, Dtype dtype,
          std::function<void(allocator::Buffer)> deleter);

    /* Construct an array by copy*/
    Array(const Array &other) = default;

    /* Construct an array in the computational graph*/
    Array(Dtype dtype, std::shared_ptr<Primitive> prim,
          std::vector<Array> inputs);

    struct Data {
        allocator::Buffer buffer;
        std::function<void(allocator::Buffer)> deleter;
        Data(const allocator::Buffer &buffer,
             std::function<void(allocator::Buffer)> deleter)
            : buffer(buffer), deleter(deleter) {}
        void *ptr() {return buffer.ptr();}
        ~Data() { deleter(buffer); }
    };

    void eval();

    bool evaluated() const { return data_->buffer.ptr() != nullptr; }

    void copyBySharing(const Array& array, size_t size, size_t offset);

    struct ArrayIterator {
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = const Array;
        using reference = value_type;

        explicit ArrayIterator(const Array &arr, int idx = 0);

        reference operator*() const;

        ArrayIterator &operator+(difference_type diff) {
            idx += diff;
            return *this;
        }

        ArrayIterator &operator++() {
            idx++;
            return *this;
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

    ArrayIterator begin() const { return ArrayIterator(*this); }
    ArrayIterator end() const { return ArrayIterator(*this); }

    std::shared_ptr<Array> tracer() { return info_->tracer_; }
    std::shared_ptr<Primitive> primitive() const { return info_->prim_; }
    std::vector<Array> &inputs() { return info_->inputs_; }

    std::vector<int> shape() const { return *(shape_); }
    Dtype dtype() const {return dtype_;}
    size_t ndim() const { return shape_->size(); }

  protected:
    std::shared_ptr<Data> data_;
    Dtype dtype_;
    size_t size_;
    std::shared_ptr<MetaData> info_;
    std::shared_ptr<std::vector<int>> shape_;

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
Array makeArrayFromScalar(int val);

} // namespace ainl::core
