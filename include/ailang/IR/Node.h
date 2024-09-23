#pragma once

#include <array>
#include <bits/stdint-intn.h>
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
    ADD,
    ALLOCA,
    ASSIGN,
    BATCHNORM2d,
    BITAND,
    BITOR,
    BITXOR,
    BROADCAST,
    CALL,
    COMPARE,
    CONCAT,
    CONSTANT,
    CONVOLUTION,
    DIV,
    EXP,
    FLOORDIV,
    GETELEMENTPTR,
    IF,
    LOAD,
    LSHIFT,
    MAKETUPLE,
    MATMUL,
    MAXPOOL2D,
    AVGPOOL2D,
    MEAN,
    MOD,
    MUL,
    NEG,
    PARAM,
    POW,
    RELU,
    RETURN,
    RSHIFT,
    STORE,
    SUB,
    TANH,
    TRANSPOSE,
    UNZIPPING,
    UNKNOWN,
    VARIANCE,
    WHILE
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
  static NodePtr create(ARGS &&...args) {
    NodePtr Node = new NodeType(std::forward<ARGS>(args)...);
    return Node;
  }

  std::vector<ValuePtr> getOperands() const;
  ValuePtr getOperand(size_t index) const;

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

NODE_PTR_TYPE_DECL(Mul)
class Mul : public Node {
public:
  Mul(const TypePtr &nodeType, const ValuePtr &lhs, const ValuePtr &rhs);
  NodeKind kind() override { return Node::NodeKind::MUL; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;

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

NODE_PTR_TYPE_DECL(Broadcast)
class Broadcast : public Node {
public:
  Broadcast(const TypePtr &nodeType, const ValuePtr &inValue,
            std::vector<int> shape);
  NodeKind kind() override { return Node::NodeKind::BROADCAST; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;
  std::vector<int> getBroadCastShape() const { return shape; }

private:
  ValuePtr inValue;
  std::vector<int> shape;
};

NODE_PTR_TYPE_DECL(Relu)
class Relu : public Node {
public:
  Relu(const TypePtr &nodeType, const ValuePtr &inValue);
  NodeKind kind() override { return Node::NodeKind::RELU; }
  explicit operator std::string() const override;
  ValuePtr getValue() const { return inValue; }
  void accept(IRVisitor *visitor) override;
  std::vector<int> getShape();

private:
  ValuePtr inValue;
};
NODE_PTR_TYPE_DECL(Mean)
class Mean : public Node {
public:
  Mean(const TypePtr &nodeType, const ValuePtr &inValue,
       const std::vector<int64_t> &dim);
  NodeKind kind() override { return Node::NodeKind::MEAN; }
  explicit operator std::string() const override;
  ValuePtr getValue() const { return inValue; }
  void accept(IRVisitor *visitor) override;
  std::vector<int64_t> getDim() { return dim; };
  std::vector<int> getShape();

private:
  ValuePtr inValue;
  std::vector<int64_t> dim;
};
NODE_PTR_TYPE_DECL(Variance)
class Variance : public Node {
public:
  Variance(const TypePtr &nodeType, const ValuePtr &inValue,
           const std::vector<int64_t> &dim, const int ddof);
  NodeKind kind() override { return Node::NodeKind::VARIANCE; }
  explicit operator std::string() const override;
  ValuePtr getValue() const { return inValue; }
  void accept(IRVisitor *visitor) override;
  std::vector<int64_t> getDim() { return dim; }
  std::vector<int> getShape();
  int getDdof() { return ddof; }

private:
  ValuePtr inValue;
  std::vector<int64_t> dim;
  int ddof;
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
  Maxpool2d(const TypePtr &nodeType, const ValuePtr &inValue,
            const std::vector<int64_t> &window_dimensions,
            const std::vector<int64_t> &window_strides,
            const std::vector<int64_t> &base_dilations,
            const std::vector<int64_t> &window_dilations,
            const std::vector<int64_t> &padding);
  NodeKind kind() override { return Node::NodeKind::MAXPOOL2D; }
  explicit operator std::string() const override;
  ValuePtr getValue() const { return inValue; }
  void accept(IRVisitor *visitor) override;
  std::vector<std::vector<int64_t>> getArgs() const {
    return {window_dimensions, window_strides, base_dilations, window_dilations,
            padding};
  }

private:
  ValuePtr inValue;
  std::vector<int64_t> window_dimensions;
  std::vector<int64_t> window_strides;
  std::vector<int64_t> base_dilations;
  std::vector<int64_t> window_dilations;
  std::vector<int64_t> padding;
};
NODE_PTR_TYPE_DECL(Avgpool2d)
class Avgpool2d : public Node {
public:
  Avgpool2d(const TypePtr &nodeType, const ValuePtr &inValue,
            const std::vector<int64_t> &window_dimensions,
            const std::vector<int64_t> &window_strides,
            const std::vector<int64_t> &base_dilations,
            const std::vector<int64_t> &window_dilations,
            const std::vector<int64_t> &padding);
  NodeKind kind() override { return Node::NodeKind::AVGPOOL2D; }
  explicit operator std::string() const override;
  ValuePtr getValue() const { return inValue; }
  void accept(IRVisitor *visitor) override;
  std::vector<std::vector<int64_t>> getArgs() const {
    return {window_dimensions, window_strides, base_dilations, window_dilations,
            padding};
  }

private:
  ValuePtr inValue;
  std::vector<int64_t> window_dimensions;
  std::vector<int64_t> window_strides;
  std::vector<int64_t> base_dilations;
  std::vector<int64_t> window_dilations;
  std::vector<int64_t> padding;
};

NODE_PTR_TYPE_DECL(Convolution)
class Convolution : public Node {
public:
  Convolution(const TypePtr &nodeType, const ValuePtr &inputValue,
              const ValuePtr &weightValue,
              const std::vector<int64_t> &window_strides,
              const std::vector<int64_t> &lhsDilation,
              const std::vector<int64_t> &rhsDilation,
              const std::vector<int64_t> &padding_args,
              const std::vector<int64_t> &window_reversal);
  NodeKind kind() override { return Node::NodeKind::CONVOLUTION; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;
  ValuePtr getInputValue() const { return inputValue; }
  ValuePtr getWeightValue() const { return weightValue; }
  std::vector<std::vector<int64_t>> getArgs() const {
    return {window_strides, lhsDilation, rhsDilation, padding_args,
            window_reversal};
  }

private:
  ValuePtr inputValue;
  ValuePtr weightValue;
  std::vector<int64_t> window_strides;
  std::vector<int64_t> lhsDilation;
  std::vector<int64_t> rhsDilation;
  std::vector<int64_t> padding_args;
  std::vector<int64_t> window_reversal;
};

NODE_PTR_TYPE_DECL(BatchNorm2d)
class BatchNorm2d : public Node {
public:
  BatchNorm2d(const TypePtr &nodeType, const ValuePtr &inValue,
              const ValuePtr &scale, const ValuePtr &offset,
              const ValuePtr &mean, const ValuePtr &variance);
  NodeKind kind() override { return Node::NodeKind::BATCHNORM2d; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;
  ValuePtr getValue() const { return inValue; }
  ValuePtr getMean() const { return mean; }
  ValuePtr getScale() const { return scale; }
  ValuePtr getVariance() const { return variance; }
  ValuePtr getOffset() const { return offset; }

private:
  ValuePtr inValue;
  ValuePtr scale;
  ValuePtr offset;
  ValuePtr mean;
  ValuePtr variance;
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

NODE_PTR_TYPE_DECL(Concat)
class Concat : public Node {
public:
  Concat(const TypePtr &nodeType, const std::vector<ValuePtr> &inputs, int dim);
  NodeKind kind() override { return Node::NodeKind::CONCAT; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;
  std::vector<ValuePtr> getInputs() { return inputs; }
  int getDim() { return dim; }

private:
  std::vector<ValuePtr> inputs;
  int dim;
};

NODE_PTR_TYPE_DECL(ConstantDef)
class ConstantDef : public Node {
public:
  ConstantDef(const TypePtr &nodeType, const ValuePtr &value);
  NodeKind kind() override { return Node::NodeKind::CONSTANT; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;

private:
  ValuePtr value;
};

NODE_PTR_TYPE_DECL(Exp)
class Exp : public Node {
public:
  Exp(const TypePtr &nodeType, const ValuePtr &inValue);
  NodeKind kind() override { return Node::NodeKind::EXP; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;

private:
  ValuePtr inValue;
};

NODE_PTR_TYPE_DECL(Tanh)
class Tanh : public Node {
public:
  Tanh(const TypePtr &nodeType, const ValuePtr &inValue);
  NodeKind kind() override { return Node::NodeKind::TANH; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;

private:
  ValuePtr inValue;
};

NODE_PTR_TYPE_DECL(Div)
class Div : public Node {
public:
  Div(const TypePtr &nodeType, const ValuePtr &lhs, const ValuePtr &rhs);
  NodeKind kind() override { return Node::NodeKind::DIV; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;

private:
  ValuePtr lhs;
  ValuePtr rhs;
};

NODE_PTR_TYPE_DECL(Neg)
class Neg : public Node {
public:
  Neg(const TypePtr &nodeType, const ValuePtr &inValue);
  NodeKind kind() override { return Node::NodeKind::NEG; }
  void accept(IRVisitor *visitor) override;
  explicit operator std::string() const override;

private:
  ValuePtr inValue;
};

} // namespace ainl::ir