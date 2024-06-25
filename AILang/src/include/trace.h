#pragma once

#include <functional>
#include <map>
#include <memory>
#include <stack>

#include "ir/function.h"

namespace ainl::core {

class Primitive;
class Tracer;
class Array;

class BaseTrace {
public:
  enum class TraceMode {
    eval,
    jit,
    jvp,
  };
  BaseTrace(int level, TraceMode mode);
  BaseTrace(const BaseTrace &other) = delete;
  BaseTrace(BaseTrace &&other) = delete;
  BaseTrace &operator=(const BaseTrace &other) = delete;
  BaseTrace &operator=(BaseTrace &&other) = delete;
  static void enableJITEagerEval();
  static void disableJITEagerEval();
  virtual ~BaseTrace() = default;
  virtual void pack(std::vector<std::shared_ptr<Tracer>> &inputs) = 0;
  virtual void unpack(std::vector<std::shared_ptr<Tracer>> &inputs) = 0;
  virtual void process(const std::shared_ptr<Primitive> &prim,
                       const std::vector<std::shared_ptr<Tracer>> &inputs,
                       const std::vector<std::shared_ptr<Tracer>> &output) = 0;
  virtual std::string toString() const = 0;

public:
  size_t level;
  TraceMode mode;
};

class EvaluationTrace : public BaseTrace {
public:
  EvaluationTrace(int level);
  void pack(std::vector<std::shared_ptr<Tracer>> &inputs);
  void unpack(std::vector<std::shared_ptr<Tracer>> &inputs);
  void process(const std::shared_ptr<Primitive> &prim,
               const std::vector<std::shared_ptr<Tracer>> &inputs,
               const std::vector<std::shared_ptr<Tracer>> &output);
  std::string toString() const override;

private:
  void update(const std::vector<std::shared_ptr<Tracer>> &inputs,
              const std::vector<Array> &output);
};

class TraceManager {
public:
  TraceManager();
  std::shared_ptr<BaseTrace> popLastTrace() {
    auto trace = traceStack.top();
    traceStack.pop();
    return trace;
  }
  void pushTrace(std::shared_ptr<BaseTrace> trace) {
    traceStack.push(std::move(trace));
  }
  std::shared_ptr<BaseTrace> getCurrentTrace() { return traceStack.top(); }
  bool hasRemainingTrace() { return traceStack.size() != 1; }

  size_t getStackSize() { return traceStack.size(); }

private:
  std::stack<std::shared_ptr<BaseTrace>> traceStack;
};

std::shared_ptr<BaseTrace> popLastTrace();
void pushTrace(std::shared_ptr<BaseTrace> trace);
std::shared_ptr<BaseTrace> getCurrentTrace();
ir::ModulePtr getTracedModule();
size_t getTraceStackSize();
std::shared_ptr<BaseTrace>
findTopTrace(const std::vector<std::shared_ptr<Tracer>> &inputs);

} // namespace ainl::core