#ifndef AINL_SRC_INCLUDE_FUNCTION_H
#define AINL_SRC_INCLUDE_FUNCTION_H

#include <iostream>
#include <memory>
#include <sstream>
#include <utility>

#include "graph.h"
#include "type.h"

class Graph;
using GraphPtr = std::shared_ptr<Graph>;

class Signature {
  public:
    Signature(TypePtr inputType, TypePtr returnType)
        : inputType(std::move(inputType)), returnType(std::move(returnType)) {}
    bool match(const Signature &rhs) {
        return (inputType->equals(*rhs.inputType) &&
                returnType->equals(*rhs.returnType));
    }
    explicit operator std::string() const {
        std::stringstream ssm;
        ssm << "(";
        ssm << inputType->getName() << ") -> ";
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

class Method : public std::enable_shared_from_this<Method> {
  private:
    SignaturePtr signature;
    GraphPtr graph;
};
using MethodPtr = std::shared_ptr<Method>;

#endif // AINL_SRC_INCLUDE_FUNCTION_H
