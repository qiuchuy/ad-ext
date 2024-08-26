#pragma once

#include <functional>
#include <map>
#include <stdexcept>

#include "ailang/IR/Function.h"
#include "ailang/IR/Type.h"

namespace ainl::ir {

ValuePtr reluNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                          const ValuePtr &inValue);
ValuePtr transposeNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                               const ValuePtr &inValue);
ValuePtr matmulNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                            const ValuePtr &lhs, const ValuePtr &rhs);
ValuePtr maxpool2dNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                               const ValuePtr &inValue);
ValuePtr convolutionNodeContract(const ModulePtr &module,
                                 const TypePtr &nodeType,
                                 const ValuePtr &inValue);
ValuePtr batchnorm2dNodeContract(const ModulePtr &module,
                                 const TypePtr &nodeType,
                                 const ValuePtr &inValue);
ValuePtr whileLoopNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                               const ModulePtr &condGraph,
                               const ModulePtr &bodyGraph,
                               std::vector<ValuePtr> args);
ValuePtr compareNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                             const ValuePtr &lhs, const ValuePtr &rhs,
                             CompareOp::CompareType op);
ValuePtr ifNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                        const ModulePtr &trueModule,
                        const ModulePtr &falseModule, const ValuePtr &cond);

class NodeContract {
public:
  explicit NodeContract();
  using AnyFunction = std::function<ValuePtr(ModulePtr, TypePtr nodeType,
                                             std::vector<ValuePtr>)>;

  void registerContract(const std::string &name, AnyFunction func) {
    functions[name] = std::move(func);
  }

  ValuePtr resolveContract(const std::string &name, ModulePtr module,
                           TypePtr nodeType, std::vector<ValuePtr> args) {
    return functions[name](std::move(module), std::move(nodeType),
                           std::move(args));
  }

private:
  std::map<std::string, AnyFunction> functions;
};

NodeContract &getNodeContract();
ValuePtr resolveContract(const std::string &name, ModulePtr module,
                         TypePtr nodeType, std::vector<ValuePtr> args);

} // namespace ainl::ir
