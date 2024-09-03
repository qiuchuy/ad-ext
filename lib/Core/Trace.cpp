#include "ailang/Core/Trace.h"

#include "ailang/Core/Array.h"
#include "ailang/Core/Primitive.h"
#include "ailang/Core/Transformation.h"
#include "ailang/IR/Function.h"
#include "ailang/Utils/Logger.h"
#include <memory>

namespace ainl::core {

BaseTrace::BaseTrace(int level, TraceMode mode) : level(level), mode(mode) {}

void BaseTrace::enableJITEagerEval() { JITTracer::eager_ = true; }
void BaseTrace::disableJITEagerEval() { JITTracer::eager_ = false; }

EvaluationTrace::EvaluationTrace(int level)
    : BaseTrace(level, BaseTrace::TraceMode::eval) {}

void EvaluationTrace::pack(std::vector<std::shared_ptr<Tracer>> &inputs) {}

void EvaluationTrace::unpack(std::vector<std::shared_ptr<Tracer>> &inputs) {}

void EvaluationTrace::process(
    const std::shared_ptr<Primitive> &prim,
    const std::vector<std::shared_ptr<Tracer>> &inputs,
    const std::vector<std::shared_ptr<Tracer>> &outputs) {
  auto outputArrays = convertTracerVector<Array>(outputs);
  prim->eval(convertTracerVector<Array>(inputs), outputArrays);
  // in order not to change the interface of `prim->eval`
  // we need to perform the update here
  // another options is to change the interface of `prim->eval` with pointer to
  // avoid a object copy
  update(outputs, outputArrays);
}

std::string EvaluationTrace::toString() const { return "eval"; }

void EvaluationTrace::update(const std::vector<std::shared_ptr<Tracer>> &inputs,
                             const std::vector<Array> &output) {
  for (size_t i = 0; i < inputs.size(); i++) {
    if (auto array = std::dynamic_pointer_cast<Array>(inputs[i])) {
      *array = output[i];
    } else {
      throw std::runtime_error(
          "Input is not an array when updating in evaluation trace.");
    }
  }
}

TraceManager::TraceManager() {
  auto evalTrace = std::make_shared<EvaluationTrace>(0);
  traceStack.push(std::dynamic_pointer_cast<BaseTrace>(evalTrace));
}

TraceManager &traceManager() {
  static TraceManager manager;
  return manager;
}

std::shared_ptr<BaseTrace> popLastTrace() {
  return traceManager().popLastTrace();
}

void pushTrace(std::shared_ptr<BaseTrace> trace) {
  traceManager().pushTrace(std::move(trace));
}

std::shared_ptr<BaseTrace> getCurrentTrace() {
  return traceManager().getCurrentTrace();
}

std::shared_ptr<BaseTrace> getStandardEvalTrace() {
  return std::make_shared<EvaluationTrace>(0);
}

ir::ModulePtr getTracedModule() {
  auto trace = getCurrentTrace();
  if (auto jit = std::dynamic_pointer_cast<JITTrace>(trace)) {
    return jit->module();
  } else {
    throw std::runtime_error(
        "[jit] Attempt to get module when running a non-JIT trace.");
  }
}

size_t getTraceStackSize() { return traceManager().getStackSize(); }

std::shared_ptr<BaseTrace>
findTopTrace(const std::vector<std::shared_ptr<Tracer>> &tracers) {
  if (tracers.size() == 0) {
    return getCurrentTrace();
  }
  std::vector<size_t> levels;
  for (const auto &tracer : tracers) {
    levels.push_back(tracer->trace()->level);
  }

  auto maxIndex = std::distance(levels.begin(),
                                std::max_element(levels.begin(), levels.end()));

  return tracers[maxIndex]->trace();
}
} // namespace ainl::core