
#include <memory>

#include "ailang/Core/Array.h"
#include "ailang/Core/Trace.h"
#include "ailang/Core/Transformation.h"
#include "ailang/IR/Container.h"
#include "ailang/IR/Function.h"
#include "ailang/IR/Literal.h"
#include "ailang/IR/Node.h"
#include "ailang/IR/Type.h"
#include "ailang/Transforms/Autodiff.h"

namespace ainl::core {

bool JVPTracer::evaluated() const {
  return primal_ != nullptr && tangent_ != nullptr;
}

std::shared_ptr<Tracer> JVPTracer::aval() {
  if (primal_ == nullptr) {
    throw std::runtime_error("Primal is not set");
  }
  return primal_->aval();
}

ir::TypePtr JVPTracer::getJITType() {
  if (primal_ == nullptr) {
    throw std::runtime_error("Primal is not set");
  }
  return primal_->getJITType();
}

std::string JVPTracer::toString() const { return "jvptracer"; }

void JVPTrace::pack(std::vector<std::shared_ptr<Tracer>> &inputs) {
  /*
  for (auto &input : inputs) {
  auto array = std::dynamic_pointer_cast<Array>(input);
  if (array) {
    input = std::make_shared<JVPTracer>(
        input,
        std::make_shared<Array>(
            fill(array->shape(), Array(1.0f, array->dtype()), array->dtype())));
  }
  }
  */
}

void JVPTrace::unpack(std::vector<std::shared_ptr<Tracer>> &inputs) {}
void JVPTrace::process(const std::shared_ptr<Primitive> &prim,
                       const std::vector<std::shared_ptr<Tracer>> &inputs,
                       const std::vector<std::shared_ptr<Tracer>> &outputs) {
  auto arrays = convertTracerVector<JVPTracer>(inputs);
  auto outputTracers = convertTracerVector<JVPTracer>(outputs);
  prim->jvp(arrays, outputTracers);
  update(outputs, outputTracers);
}

std::string JVPTrace::toString() const { return "jvp"; }

void JVPTrace::update(const std::vector<std::shared_ptr<Tracer>> &inputs,
                      const std::vector<JVPTracer> &output) {
  for (size_t i = 0; i < inputs.size(); i++) {
    if (auto array = std::dynamic_pointer_cast<JVPTracer>(inputs[i])) {
      *array = output[i];
    } else {
      throw std::runtime_error(
          "Input is not an array when updating in JVP trace.");
    }
  }
}

std::shared_ptr<Tracer>
jvp(std::function<std::shared_ptr<Tracer>(std::vector<std::shared_ptr<Tracer>>)>
        f,
    std::vector<std::shared_ptr<Tracer>> primals,
    std::vector<std::shared_ptr<Tracer>> tangents) {
  if (primals.size() != tangents.size()) {
    throw std::runtime_error("Number of primals and tangents must match");
  }
  pushTrace(std::make_shared<JVPTrace>(getTraceStackSize()));

  size_t inputSize = primals.size();

  std::vector<std::shared_ptr<Tracer>> jvpTracers;
  for (size_t i = 0; i < inputSize; i++) {
    jvpTracers.push_back(std::make_shared<JVPTracer>(primals[i], tangents[i]));
  }
  auto result = f(jvpTracers);
  result->eval();
  popLastTrace();
  return result;
}

bool JITTracer::evaluated() const { return tracer_ != nullptr; }

bool JITTracer::eager_ = true;

ir::TypePtr JITTracer::getJITType() {
  if (tracer_ == nullptr) {
    throw std::runtime_error(
        "Trying to get the jit type of an empty tracer inside JITTracer");
  }
  return tracer_->getJITType();
}

std::shared_ptr<Tracer> JITTracer::aval() {
  if (tracer_ == nullptr) {
    throw std::runtime_error(
        "Trying to get the aval of an empty tracer inside JITTracer");
  }
  return tracer_->aval();
}

std::string JITTracer::toString() const { return "jittracer"; }

void JITTrace::pack(std::vector<std::shared_ptr<Tracer>> &inputs) {
  for (auto &input : inputs) {
    ir::Value *value;
    if (!asTracer<JITTracer>(input)) {
      LOG_DEBUG("%s", "[jit] wrapping inputs with JITTracer");
      auto type = input->getJITType();
      if (!type->isTensorType()) {
        // as `getJITType` will evaluate the inner tracer
        // we can safely assume that the input is a literal when the type is not
        // a tensor type (todo) find a more precise way way to do this?
        auto array = asTracer<Array>(input->aval());
        switch (type->kind()) {
        case ir::Type::TypeKind::FloatType:
          value = ir::Literal::create(array->item<float>());
          break;
        case ir::Type::TypeKind::IntType:
          value = ir::Literal::create(array->item<int32_t>());
          break;
        default:
          throw std::runtime_error("Unsupported data type in JIT trace");
        }
      } else {
        auto ArrayTracer = asTracer<Array>(input->aval());
        auto JITTensorType = asType<TensorType>(type);
        auto ElementType = JITTensorType->getElementType();
        if (ElementType->isIntType()) {
          std::vector<ValuePtr> IntValues;
          for (size_t Offset = 0;
               Offset < ArrayTracer->size() / ArrayTracer->itemsize();
               ++Offset) {
            auto Literal = ir::Literal::create(ArrayTracer->at<int>(Offset));
            IntValues.push_back(Literal);
          }
          auto Container = TupleContainer::create(IntValues);
          value = getTracedModule()->create<ir::ConstantDef>(
              TensorType::create(ElementType, JITTensorType->getShape()),
              Container);
        } else if (ElementType->isFloatType()) {
          std::vector<ValuePtr> FloatValues;
          for (size_t Offset = 0;
               Offset < ArrayTracer->size() / ArrayTracer->itemsize();
               ++Offset) {
            auto Literal = ir::Literal::create(ArrayTracer->at<float>(Offset));
            FloatValues.push_back(Literal);
          }
          auto Container = TupleContainer::create(FloatValues);
          value = getTracedModule()->create<ir::ConstantDef>(
              TensorType::create(ElementType, JITTensorType->getShape()),
              Container);
        } else {
          throw std::runtime_error("Unsupported data type in JIT trace");
        }
      }
      auto tracer = input->clone();
      input = JITTracer::create(tracer, value);
    }
  }
}

void JITTrace::unpack(std::vector<std::shared_ptr<Tracer>> &inputs) {
  for (auto &input : inputs) {
    if (auto array = asTracer<JITTracer>(input)) {
      input = array->tracer();
    }
  }
}

void JITTrace::process(const std::shared_ptr<Primitive> &prim,
                       const std::vector<std::shared_ptr<Tracer>> &inputs,
                       const std::vector<std::shared_ptr<Tracer>> &outputs) {
  auto PackedInputs = inputs;
  pack(PackedInputs);
  auto arrays = convertTracerVector<JITTracer>(PackedInputs);
  auto outputTracers = convertTracerVector<JITTracer>(outputs);
  prim->jit(arrays, outputTracers);
  update(outputs, outputTracers);
}

void JITTrace::update(const std::vector<std::shared_ptr<Tracer>> &inputs,
                      const std::vector<JITTracer> &output) {
  for (size_t i = 0; i < inputs.size(); i++) {
    if (auto array = std::dynamic_pointer_cast<JITTracer>(inputs[i])) {
      *array = output[i];
    } else {
      throw std::runtime_error(
          "Input is not an array when updating in JIT trace.");
    }
  }
}

std::string JITTrace::toString() const { return "jit"; }

ir::ModulePtr jit(std::function<std::vector<std::shared_ptr<Tracer>>(
                      std::vector<std::shared_ptr<Tracer>>)>
                      f,
                  std::string funcName,
                  const std::vector<std::shared_ptr<Tracer>> &inputs) {
  std::vector<ir::TypePtr> types;
  for (auto &input : inputs) {
    types.push_back(input->getJITType());
  }
  auto argType = ir::TupleType::createUnnamedTuple(types);
  auto module = ir::ALModule::create(funcName, argType);
  auto params = module->getParams();

  pushTrace(std::make_shared<JITTrace>(module, getTraceStackSize()));
  std::vector<std::shared_ptr<Tracer>> jittracers;

  for (size_t i = 0; i < params.size(); i++) {
    auto jittracer = JITTracer::create(inputs[i], params[i]);
    jittracers.push_back(jittracer);
  }

  auto result = (f(jittracers));
  if (result.size() == 0) {
    throw std::runtime_error("JIT function must return at least one value");
  }
  eval(result);
  if (result.size() > 1) {
    std::vector<ir::TypePtr> resultTypes;
    std::vector<ir::ValuePtr> resultValues;
    for (auto &input : result) {
      resultTypes.push_back(asTracer<JITTracer>(input)->value()->getType());
      resultValues.push_back(asTracer<JITTracer>(input)->value());
    }
    auto returnValue = ir::TupleContainer::create(resultValues);
    module->setReturnType(returnValue->getType());
    module->getGraph()->create<ir::ReturnOp>(returnValue);
  } else {
    auto returnValue = asTracer<JITTracer>(result[0])->value();
    module->getGraph()->create<ir::ReturnOp>(returnValue);
    module->setReturnType(returnValue->getType());
  }
  popLastTrace();
  return module;
}

ir::ModulePtr grad(std::function<std::vector<std::shared_ptr<Tracer>>(
                       std::vector<std::shared_ptr<Tracer>>)>
                       f,
                   std::string FuncName,
                   const std::vector<std::shared_ptr<Tracer>> &Inputs) {
  auto Module = jit(f, "d" + FuncName, Inputs);
  ir::autodiffOnModule(Module);
  return Module;
}

void eval(const std::vector<std::shared_ptr<Tracer>> &inputs) {
  for (auto &input : inputs) {
    // inputs should be siblings on computation graph
    // so actually this loop will only be executed once
    input->eval();
  }
}

} // namespace ainl::core