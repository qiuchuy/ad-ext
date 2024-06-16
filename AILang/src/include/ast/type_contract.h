#pragma once

#include "ir/symbol.h"
#include "ir/type.h"
#include "utils/utils.h"
#include <stdexcept>

namespace ainl::ir {

TypePtr matmulTypeContract(const TypePtr &lhsType, const TypePtr &rhsType);
TypePtr transposeTypeContract(const TypePtr &inType);
TypePtr addTypeContract(const TypePtr &lhsType, const TypePtr &rhsType);
TypePtr reluTypeContract(const TypePtr &inType);
TypePtr maxpool2dTypeContract(const TypePtr &inType);
TypePtr convolutionTypeContract(const TypePtr &inType);
TypePtr batchnorm2dTypeContract(const TypePtr &inType);
TypePtr compareTypeContract(const TypePtr &lhsType, const TypePtr &rhsType);

class TypeContract {
public:
  explicit TypeContract();
  using AnyFunction = std::function<TypePtr(std::vector<TypePtr>)>;

  void registerContract(const std::string &name, AnyFunction func) {
    functions[name] = std::move(func);
  }

  TypePtr resolveContract(const std::string &name, std::vector<TypePtr> args) {
    if (functions.find(name) == functions.end()) {
      // throw AINLError(
      // "This operator has not been registered into the library yet.");
      throw std::runtime_error(
          "This operator has not been registered into the library yet.");
    }
    return functions[name](std::move(args));
  }

private:
  std::map<std::string, AnyFunction> functions;
};

TypeContract &getTypeContract();
TypePtr resolveContract(const std::string &name, std::vector<TypePtr> args);

} // namespace ainl::ir