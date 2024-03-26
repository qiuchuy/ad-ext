#pragma once

#include <functional>
#include <map>
#include <memory>
#include <stack>

#include "array.h"
#include "primitive.h"

namespace ainl::core {

class Primitive;

class BaseTrace : public std::enable_shared_from_this<BaseTrace> {
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
  virtual void pack(Array &inputs) = 0;
  virtual void unpack(Array &outputs) = 0;
  virtual void process(const std::shared_ptr<Primitive> &prim,
                       std::vector<Array> &inputs, Array &output) = 0;
};

class EvaluationTrace : public BaseTrace {
public:
  EvaluationTrace();
  virtual void pack(Array &array);
  virtual void unpack(Array &array);
  virtual void process(const std::shared_ptr<Primitive> &prim,
                       std::vector<Array> &inputs, Array &output);
};

class JITTrace : public BaseTrace {
public:
  JITTrace();
  virtual void pack(Array &array);
  virtual void unpack(Array &array);
  virtual void process(const std::shared_ptr<Primitive> &prim,
                       std::vector<Array> &inputs, Array &output);
};

class TraceManager {
public:
  TraceManager();
  std::shared_ptr<BaseTrace> popLastTrace() {
    auto trace = traceStack.top();
    traceStack.pop();
    return trace;
  }
  std::shared_ptr<BaseTrace> getCurrentTrace() { return traceStack.top(); }
  bool hasRemainingTrace() { return !traceStack.empty(); }

private:
  std::stack<std::shared_ptr<BaseTrace>> traceStack;
};

TraceManager &traceManager();
std::shared_ptr<BaseTrace> popLastTrace();
std::shared_ptr<BaseTrace> getCurrentTrace();
bool hasRemainingTrace();

} // namespace ainl::core