
#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <utility>

#include "ir/graph.h"
#include "ir/type.h"

namespace ainl::ir {

class Signature {
  public:
    Signature() = default;
    Signature(TypePtr inputType, TypePtr returnType)
        : inputType(std::move(inputType)), returnType(std::move(returnType)) {}
    bool match(const Signature &rhs) {
        return (inputType->equals(*rhs.inputType) &&
                returnType->equals(*rhs.returnType));
    }
    explicit operator std::string() const {
        std::stringstream ssm;
        ssm << inputType->getName() << " -> ";
        ssm << returnType->getName();
        return ssm.str();
    }
    friend class ALModule;
    friend std::ostream &operator<<(std::ostream &stream,
                                    const Signature *signature) {
        stream << std::string(*signature);
        return stream;
    }

    bool operator==(const Signature &other) { return match(other); }
    bool operator!=(const Signature &other) { return !match(other); }

  private:
    TypePtr inputType;
    TypePtr returnType;
};
using SignaturePtr = Signature *;

class ALModule;
using ModulePtr = std::shared_ptr<ALModule>;
class ALModule : public std::enable_shared_from_this<ALModule> {
  public:
    ALModule() = default;
    ALModule(std::string name, const TypePtr &inputType,
             const TypePtr &returnType = nullptr);
    static ModulePtr create(std::string name, const TypePtr &inputType,
                            const TypePtr &returnType = nullptr) {
        return std::make_shared<ALModule>(name, inputType, returnType);
    }
    std::vector<ValuePtr> getParams();
    std::vector<TypePtr> getParamTypes();
    std::vector<TypePtr> getReturnTypes();
    void setReturnType(const TypePtr &returnType) {
        signature->returnType = returnType;
    }
    GraphPtr getGraph() { return graph; }
    std::string getName() { return name; }
    std::string str();

  private:
    SignaturePtr signature;
    GraphPtr graph;
    std::string name;
};
} // namespace ainl::ir
