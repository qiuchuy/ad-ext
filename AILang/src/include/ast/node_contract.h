#pragma once

#include <functional>
#include <map>

#include "ir/function.h"

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
    if (functions.find(name) == functions.end()) {
      // throw AINLError(
      // "This operator has not been registered into the library yet.");
    }
    return functions[name](std::move(module), std::move(nodeType),
                           std::move(args));
  }

private:
  std::map<std::string, AnyFunction> functions;
};

NodeContract &getNodeContract();

} // namespace ainl::ir
