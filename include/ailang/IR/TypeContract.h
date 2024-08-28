#pragma once

#include "ailang/IR/Symbol.h"
#include "ailang/IR/Type.h"
#include "ailang/Utils/Utils.h"
#include <stdexcept>

namespace ainl::ir {

TypePtr matmulTypeContract(const TypePtr &lhsType, const TypePtr &rhsType);
TypePtr transposeTypeContract(const TypePtr &inType);
TypePtr addTypeContract(const TypePtr &lhsType, const TypePtr &rhsType);
TypePtr reluTypeContract(const TypePtr &inType);
TypePtr meanTypeContract(const TypePtr &inType);

// TypePtr broadcastTypeContract(const TypePtr &inType);
TypePtr maxpool2dTypeContract(const TypePtr &inType);
TypePtr convolutionTypeContract(const TypePtr &inputType,
                                const TypePtr &weightType);
TypePtr batchnorm2dTypeContract(const TypePtr &inType);
TypePtr compareTypeContract(const TypePtr &lhsType, const TypePtr &rhsType);
TypePtr ifTypeContract(const TypePtr &condType, const TypePtr &trueType,
                       const TypePtr &falseType);

class TypeContract {
  public:
    explicit TypeContract();
    using AnyFunction = std::function<TypePtr(std::vector<TypePtr>)>;

    void registerContract(const std::string &name, AnyFunction func) {
        functions[name] = std::move(func);
    }

    TypePtr resolveContract(const std::string &name,
                            std::vector<TypePtr> args) {
        if (functions.find(name) == functions.end()) {
            throw std::runtime_error("The type contract of operator [" + name +
                                     "] has not been registered yet.");
        }
        return functions[name](std::move(args));
    }

  private:
    std::map<std::string, AnyFunction> functions;
};

TypeContract &getTypeContract();
TypePtr resolveContract(const std::string &name, std::vector<TypePtr> args);

} // namespace ainl::ir