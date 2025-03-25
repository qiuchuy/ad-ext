#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <utility>

#include "ailang/IR/Container.h"
#include "ailang/IR/Graph.h"
#include "ailang/IR/Literal.h"
#include "ailang/IR/Type.h"

namespace ainl::ir {

class Signature {
public:
  Signature() = default;
  Signature(TypePtr inputType, TypePtr returnType)
      : inputType(std::move(inputType)), returnType(std::move(returnType)) {}
  bool match(const Signature &rhs) {
    return (inputType->equals(rhs.inputType) &&
            returnType->equals(rhs.returnType));
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

class Graph;

class ALModule;
using ModulePtr = std::shared_ptr<ALModule>;
class ALModule : public std::enable_shared_from_this<ALModule>, public Value {
public:
  ALModule() = default;
  ALModule(std::string name, const TypePtr &inputType,
           const TypePtr &returnType = nullptr);
  static ModulePtr create(std::string name, const TypePtr &inputType,
                          const TypePtr &returnType = nullptr) {
    return std::make_shared<ALModule>(name, inputType, returnType);
  }
  static ALModule *createModuleValue(const ALModule &module) {
    return new ALModule(module);
  }
  static ModulePtr create(const TypePtr &inputType,
                          const TypePtr &returnType = nullptr) {
    return std::make_shared<ALModule>("", inputType, returnType);
  }
  template <typename NodeType, typename... ARGS>
  NodePtr create(ARGS &&... args) {
    NodePtr Node = new NodeType(std::forward<ARGS>(args)...);
    Node->graph = graph;
    Node->block = (BlockPtr)(graph->endBlock->prev);
    graph->insertNodeAtEnd(Node);
    return Node;
  }

  template <typename NodeType, typename... ARGS>
  NodePtr createAfter(NodePtr after, ARGS &&... args) {
    NodePtr Node = new NodeType(std::forward<ARGS>(args)...);
    Node->graph = graph;
    Node->block = after->block;
    graph->insertNodeAfter(after, Node);
    return Node;
  }

  template <typename ConstantType>
  ValuePtr createConstantValue(const ConstantType &Constant, TypePtr Type) {
    auto ValueTensorType = asType<TensorType>(Type);
    auto Shape = ValueTensorType->getConcreteShape();
    std::vector<ValuePtr> FloatValues;
    if (Shape.empty()) {
      FloatValues.push_back(Literal::create(Constant));
    } else {
      size_t NumElements = 1;
      for (auto Axis : Shape) {
        NumElements *= Axis;
      }
      for (size_t Idx = 0; Idx < NumElements; Idx++) {
        FloatValues.push_back(Literal::create(Constant));
      }
    }
    return create<ConstantDef>(Type, TupleContainer::create(FloatValues));
  }

  void remove(NodePtr Node) { graph->remove(Node); }
  std::vector<ValuePtr> getParams();
  std::vector<TypePtr> getParamTypes();
  TypePtr getReturnType();
  void setReturnType(const TypePtr &returnType) {
    signature->returnType = returnType;
  }
  GraphPtr getGraph() { return graph; }
  std::string getName() { return name; }
  void setName(const std::string& new_name) { name = new_name;}
  std::string str();
  explicit operator std::string() { return str(); }

private:
  SignaturePtr signature;
  GraphPtr graph;
  std::string name;
};
} // namespace ainl::ir
