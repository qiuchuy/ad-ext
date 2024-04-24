#pragma once

#include <functional>
#include <map>
#include <memory>
#include <stack>

namespace ainl::core {

class Primitive;
class Tracer;

class BaseTrace {
public:
  enum class TraceMode {
    eval,
    symbol,
  };
  BaseTrace() = default;
  BaseTrace(const BaseTrace &other) = delete;
  BaseTrace(BaseTrace &&other) = delete;
  BaseTrace &operator=(const BaseTrace &other) = delete;
  BaseTrace &operator=(BaseTrace &&other) = delete;
  virtual ~BaseTrace() = default;
  virtual void pack(std::vector<std::shared_ptr<Tracer>> &inputs) = 0;
  virtual void unpack(std::vector<std::shared_ptr<Tracer>> &inputs) = 0;
  virtual void process(const std::shared_ptr<Primitive> &prim,
                       const std::vector<std::shared_ptr<Tracer>> &inputs,
                       std::shared_ptr<Tracer> &output) = 0;
  virtual std::string toString() const = 0;
};

class EvaluationTrace : public BaseTrace {
public:
  EvaluationTrace();
  void pack(std::vector<std::shared_ptr<Tracer>> &inputs);
  void unpack(std::vector<std::shared_ptr<Tracer>> &inputs);
  void process(const std::shared_ptr<Primitive> &prim,
               const std::vector<std::shared_ptr<Tracer>> &inputs,
               std::shared_ptr<Tracer> &output);
  std::string toString() const override;
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

private:
  std::stack<std::shared_ptr<BaseTrace>> traceStack;
};

TraceManager &traceManager();
std::shared_ptr<BaseTrace> popLastTrace();
void pushTrace(std::shared_ptr<BaseTrace> trace);
std::shared_ptr<BaseTrace> getCurrentTrace();

} // namespace ainl::core