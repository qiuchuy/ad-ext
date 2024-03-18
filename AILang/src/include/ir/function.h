#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <utility>

#include "ir/graph.h"
#include "ir/type.h"

// class Graph;
// using GraphPtr = std::shared_ptr<Graph>;
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
             const TypePtr &returnType);
    std::vector<ValuePtr> getParams();
    GraphPtr getGraph() { return graph; }
    std::string getName() { return name; }
    std::string str();

  private:
    SignaturePtr signature;
    GraphPtr graph;
    std::string name;
};
} // namespace ainl::ir
