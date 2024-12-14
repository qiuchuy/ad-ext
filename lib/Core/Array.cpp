#include <numeric>
#include <sstream>
#include <stdexcept>

#include "ailang/Core/Array.h"
#include "ailang/Core/Graph.h"
#include "ailang/Core/Ops.h"
#include "ailang/Core/Primitive.h"
#include "ailang/Core/Trace.h"
#include "ailang/Core/Transformation.h"
#include "ailang/IR/Literal.h"
#include "ailang/IR/Type.h"
#include "ailang/Utils/Logger.h"

namespace ainl::core {

Array::Array(Dtype dtype, std::shared_ptr<Primitive> prim,
             const std::vector<Array> &inputs, const std::vector<int> &shape,
             const std::vector<int> &stride)
    : Tracer({}, prim), device_(cpu) {
  std::vector<std::shared_ptr<Tracer>> inputTracers;
  for (const auto &input : inputs) {
    inputTracers.push_back(
        std::dynamic_pointer_cast<Tracer>(std::make_shared<Array>(input)));
  }
  dtype_ = dtype;
  inputs_ = inputTracers;
  shape_ = std::make_shared<std::vector<int>>(shape);
  stride_ = std::make_shared<std::vector<int>>(stride);
  trace_ = getStandardEvalTrace();
  idx_ = 0;
  size_ =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
      dtypeSize(dtype);
}

Array::Array(const std::vector<std::shared_ptr<Tracer>> &inputs,
             const std::shared_ptr<Primitive> &prim)
    : Tracer(inputs, prim) {
  if (inputs.size()) {
    auto input = asTracer<Array>(inputs[0]);
    device_ = input->device();
  }
  trace_ = getStandardEvalTrace();
}

Array::Array(const allocator::Buffer &buffer, Dtype dtype,
             const std::vector<int> &shape, const std::vector<int> &stride,
             Device device)
    : Tracer({}, std::make_shared<IdentityPrimitive>()),
      data_(std::make_shared<Data>(
          buffer, [](allocator::Buffer buffer) { allocator::free(buffer); })),
      dtype_(dtype), device_(device),
      shape_(std::make_shared<std::vector<int>>(shape)),
      stride_(std::make_shared<std::vector<int>>(stride)) {
  ptr_ = buffer.ptr();
  trace_ = getStandardEvalTrace();
  size_ =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
      dtypeSize(dtype);
}

Tracer::Tracer(const std::vector<std::shared_ptr<Tracer>> &inputs,
               const std::shared_ptr<Primitive> &prim)
    : inputs_(inputs), prim_(prim), trace_(getCurrentTrace()), idx_(0) {}

Tracer::Tracer(const std::vector<std::shared_ptr<Tracer>> &inputs,
               const std::shared_ptr<Primitive> &prim, uint64_t idx)
    : inputs_(inputs), prim_(prim), trace_(getCurrentTrace()), idx_(idx) {}

void Tracer::eval() {
  LOG_DEBUG("%s", "Starting evaluating tracers as a subgraph.");
  std::function<void(std::shared_ptr<Tracer> tracer)> recursion =
      [&recursion](std::shared_ptr<Tracer> tracer) -> void {
    auto trace = findTopTrace(tracer->inputs());
    LOG_DEBUG("%s", std::string("[eval] Current program transformation: " +
                                trace->toString())
                        .c_str());
    if (tracer->evaluated()) {
      return;
    } else {
      for (auto &input : tracer->inputs()) {
        recursion(input);
      }
      if (!tracer->isLeaf()) {
        LOG_DEBUG("[eval] Evaluating tracer with primitive %s",
                  tracer->primitive()->toString().c_str());
        trace->process(tracer->primitive(), tracer->inputs(),
                       tracer->outputs());
      }
    }
  };

  recursion(shared_from_this());
}

bool Tracer::evaluated() const { return false; }

std::string Tracer::toString() const { return "tracer"; }

bool Tracer::operator==(Tracer &other) {
  if (auto array = asTracer<Array>(shared_from_this())) {
    if (auto another = asTracer<Array>(other.shared_from_this())) {
      return array->operator==(*another);
    } else {

      return aval() == other.aval();
    }
  }
}

bool Tracer::operator>(Tracer &other) {
  if (auto array = asTracer<Array>(shared_from_this())) {
    if (auto another = asTracer<Array>(other.shared_from_this())) {
      return array->operator>(*another);
    } else {
      return aval() == other.aval();
    }
  }
}

void Array::copyBySharing(const Array &other, size_t size, size_t offset,
                          const std::vector<int> &shape,
                          const std::vector<int> &stride) {
  data_ = other.data_;
  ptr_ = (char *)other.ptr_ + offset;
  shape_ = std::make_shared<std::vector<int>>(shape);
  dtype_ = other.dtype_;
  size_ = size;
  inputs_ = other.inputs_;
  prim_ = other.prim_;
  if (stride.empty()) {
    stride_ = std::make_shared<std::vector<int>>(shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
      stride_->at(i) = 1;
      for (size_t j = i + 1; j < shape.size(); j++) {
        stride_->at(i) *= shape[j];
      }
      stride_->at(i) *= dtypeSize(dtype_);
    }
  } else {
    stride_ = std::make_shared<std::vector<int>>(stride);
  }
}

void Array::setDataWithBuffer(allocator::Buffer buffer, Dtype dtype,
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
  os<< std::endl;

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
  return oss.str();
}

ir::TypePtr Array::getJITType() {
  if (!evaluated())
    eval();
  if (shape_->empty()) {
    // tensor which has empty shape_ will be jitted into literals
    switch (dtype_.type) {
    case Dtype::DataType::BoolType:
      return ir::TensorType::create(ir::BoolTypePtr::get(), {});
    case Dtype::DataType::Int32Type:
      return ir::TensorType::create(ir::IntTypePtr::get(), {});
    case Dtype::DataType::Float32Type:
      return ir::TensorType::create(ir::FloatTypePtr::get(), {});
    default:
      throw std::invalid_argument("Unsupported jit dtype");
    }
  }
  auto eleTy = ir::DtypeToTypePtr(dtype_);
  std::vector<ir::ValuePtr> shape;
  for (const auto &dim : *shape_) {
    shape.push_back(ir::Literal::create(dim));
  }
  return ir::TensorType::create(eleTy, shape);
}

} // namespace ainl::core
