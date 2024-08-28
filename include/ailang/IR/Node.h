#pragma once

#include <array>
#include <utility>

#include "ailang/IR/Block.h"
#include "ailang/IR/Type.h"
#include "ailang/IR/Value.h"

namespace ainl::ir {

#define NODE_PTR_TYPE_DECL(name)                                               \
  class name;                                                                  \
  using name##Ptr = name *;

// Language builtin Nodes
// Extend this class to add more Nodes
class Graph;
class ALModule;
class Block;
class Signature;
class IRVisitor;
using GraphPtr = std::shared_ptr<Graph>;
using ModulePtr = std::shared_ptr<ALModule>;
using BlockPtr = Block *;
using SignaturePtr = Signature *;

NODE_PTR_TYPE_DECL(Node)
class Node : public Value {
public:
  enum class NodeKind {
    ALLOCA,
    STORE,
    LOAD,
    GETELEMENTPTR,
    ASSIGN,
    ADD,
    SUB,
    MUL,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    LSHIFT,
    RSHIFT,
    BITOR,
    BITXOR,
    BITAND,
    MATMUL,
    CALL,
    PARAM,
    RETURN,
    MAKETUPLE,
    UNZIPPING,
    RELU,
    TRANSPOSE,
    MAXPOOL2D,
    CONVOLUTION,
    BATCHNORM2d,
    WHILE,
    IF,
    UNKNOWN,
    COMPARE,
  };

  void init() {
    useList.clear();
    useValueList.clear();
    setNext(nullptr);
    setPrev(nullptr);
  }
  ~Node() override = default;
  Node();
  Node(const TypePtr &type);

  explicit operator std::string() const override { return ""; }
  static Node *create(const TypePtr &type) { return new Node(type); }
  Value::ValueKind getValueKind() const override {
    return Value::ValueKind::Node;
  }

  template <typename NodeType, typename... ARGS>
  static NodePtr create(ARGS &&... args) {
    NodePtr Node = new NodeType(std::forward<ARGS>(args)...);
    return Node;
  }

  std::vector<ValuePtr> getOperands();
  ValuePtr getOperand(size_t index);

  virtual NodeKind kind() { return Node::NodeKind::UNKNOWN; }
  virtual void accept(IRVisitor *visitor);
  virtual std::vector<ValuePtr> getOutputValues() { return {this}; }
  ValuePtr getOutputValue(size_t i) { return getOutputValues()[i]; }

  friend class Graph;

  static int LOCAL_COUNT;

  void addBlock();
  void addBlockWithParam(NodePtr param, GraphPtr graph);
  void setUse(ValuePtr value, int idx);

protected:
  std::vector<ValuePtr> useValueList;
  std::vector<UsePtr> useList;
};

NODE_PTR_TYPE_DECL(Param)
class Param : public Node {
public:
  Param(std::vector<ValuePtr> params, const TypePtr &types);
  static ParamPtr create(std::vector<ValuePtr> params, const TypePtr &types) {
    return new Param(std::move(params), types);
  }
  std::vector<ValuePtr> getParams() { return params; }
  NodeKind kind() override { return Node::NodeKind::PARAM; }
  explicit operator std::string() const override {
    std::stringstream ssm;
    ssm << "(";
    if (!params.empty()) {
      for (size_t i = 0; i < params.size() - 1; i++)
        ssm << params[i]->getName() << " : " << params[i]->getType()->str()
            << ", ";
      ssm << params[params.size() - 1]->getName() << " : "
          << params[params.size() - 1]->getType()->str();
    }
    ssm << ")";
    return ssm.str();
  }
  void accept(IRVisitor *visitor) override;
  void addParam(ValuePtr Param, const TypePtr &Type, size_t Index);

private:
  std::vector<ValuePtr> params;
  TypePtr contentType;
};

NODE_PTR_TYPE_DECL(ReturnOp)
class ReturnOp : public Node {
public:
  explicit ReturnOp(const ValuePtr &value);
  static ReturnOpPtr create(ValuePtr value) { return new ReturnOp(value); }

  NodeKind kind() override { return NodeKind::RETURN; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override {
    std::stringstream ssm;
    ssm << "return ";
    ssm << value->getName();
    return ssm.str();
  }

  ValuePtr getReturnValue() const { return value; }
  void setReturnValue(ValuePtr Value) { value = Value; }

private:
  ValuePtr value;
};

NODE_PTR_TYPE_DECL(Add)
class Add : public Node {
public:
  Add(const TypePtr &nodeType, const ValuePtr &lhs, const ValuePtr &rhs);
  NodeKind kind() override { return Node::NodeKind::ADD; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;
  ValuePtr getLHS() const { return lhs; }
  ValuePtr getRHS() const { return rhs; }

private:
  ValuePtr lhs;
  ValuePtr rhs;
};

NODE_PTR_TYPE_DECL(Matmul)
class Matmul : public Node {
public:
  Matmul(const TypePtr &nodeType, const ValuePtr &lhs, const ValuePtr &rhs);
  NodeKind kind() override { return Node::NodeKind::MATMUL; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;
  ValuePtr getLHS() const { return lhs; }
  ValuePtr getRHS() const { return rhs; }

private:
  ValuePtr lhs;
  ValuePtr rhs;
};

NODE_PTR_TYPE_DECL(Relu)
class Relu : public Node {
public:
  Relu(const TypePtr &nodeType, const ValuePtr &inValue);
  NodeKind kind() override { return Node::NodeKind::RELU; }
  explicit operator std::string() const override;
  // 在需要将 Relu 类的对象转换为字符串类型时使用。
  ValuePtr getValue() const { return inValue; }

private:
  ValuePtr inValue;
};

NODE_PTR_TYPE_DECL(Transpose)
class Transpose : public Node {
public:
  // 似乎有同名类
  Transpose(const TypePtr &nodeType, const ValuePtr &inValue);
  NodeKind kind() override { return Node::NodeKind::TRANSPOSE; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;
  // 在需要将 Transpsoe 类的对象转换为字符串类型时使用。
  ValuePtr getValue() const { return inValue; }
  std::vector<int> getShape();

private:
  ValuePtr inValue;
};

NODE_PTR_TYPE_DECL(Maxpool2d)
class Maxpool2d : public Node {
public:
  Maxpool2d(const TypePtr &nodeType, const ValuePtr &inValue);
  NodeKind kind() override { return Node::NodeKind::MAXPOOL2D; }
  explicit operator std::string() const override;
  ValuePtr getValue() const { return inValue; }

private:
  ValuePtr inValue;
};

NODE_PTR_TYPE_DECL(Convolution)
class Convolution : public Node {
public:
  Convolution(const TypePtr &nodeType, const ValuePtr &inValue);
  NodeKind kind() override { return Node::NodeKind::CONVOLUTION; }
  explicit operator std::string() const override;
  ValuePtr getValue() const { return inValue; }

private:
  ValuePtr inValue;
};

NODE_PTR_TYPE_DECL(BatchNorm2d)
class BatchNorm2d : public Node {
public:
  BatchNorm2d(const TypePtr &nodeType, const ValuePtr &inValue);
  NodeKind kind() override { return Node::NodeKind::BATCHNORM2d; }
  explicit operator std::string() const override;
  ValuePtr getValue() const { return inValue; }

private:
  ValuePtr inValue;
};

NODE_PTR_TYPE_DECL(WhileOp)
class WhileOp : public Node {
public:
  WhileOp(const TypePtr &nodeType, const ModulePtr &cond, const ModulePtr &body,
          const std::vector<ValuePtr> &inits);
  NodeKind kind() override { return Node::NodeKind::WHILE; }
  explicit operator std::string() const override;
  std::vector<ValuePtr> getOutputValues() override { return outs; }

private:
  ModulePtr cond;
  ModulePtr body;
  std::vector<ValuePtr> inits;
  std::vector<ValuePtr> outs;
};

NODE_PTR_TYPE_DECL(IfOp)
class IfOp : public Node {
public:
  IfOp(const TypePtr &nodeType, const ModulePtr &trueBranch,
       const ModulePtr &falseBranch, const ValuePtr &cond);
  NodeKind kind() override { return Node::NodeKind::IF; }
  explicit operator std::string() const override;
  void accept(IRVisitor *visitor) override;
  std::vector<ValuePtr> getOutputValues() override { return outs; }
  ValuePtr getCond() const { return cond; }
  ModulePtr getThenBranch() const { return trueBody; }
  ModulePtr getFalseBranch() const { return elseBody; }

private:
  ModulePtr trueBody;
  ModulePtr elseBody;
  ValuePtr cond;
  std::vector<ValuePtr> outs;
};

NODE_PTR_TYPE_DECL(CompareOp)
class CompareOp : public Node {
public:
  enum struct CompareType : size_t {
    EQ = 0,
    NE,
    GE,
    GT,
    LE,
    LT,
    COMPARETYPE,
  };
  CompareOp(const TypePtr &nodeType, const ValuePtr &lhs, const ValuePtr &rhs,
            CompareType compareType);
  NodeKind kind() override { return Node::NodeKind::COMPARE; }
  ValuePtr getLHS() { return lhs; }
  ValuePtr getRHS() { return rhs; }
  size_t getCompareDirection() { return static_cast<size_t>(op); }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;

private:
  ValuePtr lhs;
  ValuePtr rhs;
  CompareType op;
};

const std::array<std::string,
                 static_cast<size_t>(CompareOp::CompareType::COMPARETYPE)>
    compareOpString = {"eq", "ne", "ge", "gt", "le", "lt"};

} // namespace ainl::ir