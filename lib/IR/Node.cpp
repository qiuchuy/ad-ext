
#include <memory>
#include <utility>

#include "ailang/AST/AST.h"
#include "ailang/IR/Block.h"
#include "ailang/IR/Function.h"
#include "ailang/IR/IRVisitor.h"
#include "ailang/IR/Node.h"

namespace ainl::ir {

int Node::LOCAL_COUNT = 0;

Node::Node() { init(); }

Node::Node(const TypePtr &type) : Value(type) {
  init();
  prefix = LOCAL_PREFIX;
  name = LOCAL_NAME_PREFIX + std::to_string(LOCAL_COUNT++);
}

void Node::setUse(ValuePtr value, int idx) {
  auto use = new Use(this, value, idx);
  value->insertUseAtEnd(use);
  useList.push_back(use);
  useValueList.push_back(value);
}

void Node::addBlock() {
  auto newBlock = new Block();
  if (this->block->endBlock) {
    this->block->endBlock->insertBefore(newBlock);
    return;
  }
  this->block->beginBlock = new Block();
  this->block->endBlock = new Block();
  this->block->beginBlock->setNext(this->block->endBlock);
  this->block->endBlock->setPrev(this->block->beginBlock);
  this->block->endBlock->insertBefore(block);
}

std::vector<ValuePtr> Node::getOperands() const {
  std::vector<ValuePtr> Operands;
  for (auto &Use : useList) {
    Operands.push_back(Use->used);
  }
  return Operands;
}

ValuePtr Node::getOperand(size_t index) const {
  if (index >= useList.size()) {
    throw std::runtime_error(
        "Index out of range when getting IR Node operand.");
  }
  return useList[index]->used;
}

void Node::addBlockWithParam(NodePtr param, GraphPtr graph) {
  auto newBlock = new Block(Block::blockCount++);
  newBlock->paramNode = dynamic_cast<ParamPtr>(param);
  for (auto &innerParam : dynamic_cast<ParamPtr>(param)->getParams()) {
    innerParam->block = newBlock;
  }
  param->block = newBlock;
  if (this->block->endBlock) {
    this->block->endBlock->insertBefore(newBlock);
    return;
  }
  /* for nested blocks */
  this->block->beginBlock = new Block();
  this->block->endBlock = new Block();
  this->block->beginBlock->setNext(this->block->endBlock);
  this->block->endBlock->setPrev(this->block->beginBlock);

  /* insert this new block to graph */
  graph->endBlock->insertBefore(newBlock);
}

void Node::accept(IRVisitor *visitor) { visitor->visit(this); }

Param::Param(std::vector<ValuePtr> params, const TypePtr &type) : Node(type) {
  for (size_t i = 0; i < params.size(); i++) {
    setUse(params[i], i);
  }
  this->params = std::move(params);
  this->contentType = type;
}

void Param::accept(IRVisitor *visitor) { visitor->visit(this); }

void Param::addParam(ValuePtr param, const TypePtr &type, size_t Index) {
  setUse(param, Index);
  params.push_back(param);
}

ReturnOp::ReturnOp(const ValuePtr &value) : Node(value->getType()) {
  setUse(value, 0);
  this->value = value;
}

void ReturnOp::accept(IRVisitor *visitor) { visitor->visit(this); }

Mul::Mul(const TypePtr &opType, const ValuePtr &lhs, const ValuePtr &rhs)
    : Node(opType) {
  setUse(lhs, 0);
  setUse(rhs, 1);
  this->lhs = lhs;
  this->rhs = rhs;
}

Mul::operator std::string() const {
  return getName() + " = ailang::mul(" + getOperand(0)->getName() + ", " +
         getOperand(1)->getName() + "): " + std::string(*getType());
}

void Mul::accept(IRVisitor *visitor) { visitor->visit(this); }

// Matmul
Matmul::Matmul(const TypePtr &opType, const ValuePtr &lhs, const ValuePtr &rhs)
    : Node(opType) {
  setUse(lhs, 0);
  setUse(rhs, 1);
  this->lhs = lhs;
  this->rhs = rhs;
}

Matmul::operator std::string() const {
  return getName() + " = ailang::matmul(" + getLHS()->getName() + ", " +
         getRHS()->getName() + "): " + std::string(*getType());
}

void Matmul::accept(IRVisitor *visitor) { visitor->visit(this); }
// Add
Add::Add(const TypePtr &opType, const ValuePtr &lhs, const ValuePtr &rhs)
    : Node(opType) {
  this->lhs = lhs;
  this->rhs = rhs;
}

Add::operator std::string() const {
  return getName() + " = ailang::add(" + getLHS()->getName() + ", " +
         getRHS()->getName() + "): " + std::string(*getType());
}

void Add::accept(IRVisitor *visitor) { visitor->visit(this); }

Broadcast::Broadcast(const TypePtr &opType, const ValuePtr &inValue,
                     std::vector<int> shape)
    : Node(opType), inValue(inValue), shape(shape) {
  setUse(inValue, 0);
}

Broadcast::operator std::string() const {
  std::string prefix = getName() + " = ailang::broadcast(" +
                       getOperand(0)->getName() +
                       "):" + std::string(*getType());
  std::string postfix = "<shape=[";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i == shape.size() - 1) {
      postfix += std::to_string(shape[i]) + "]>";
    } else {
      postfix += std::to_string(shape[i]) + ",";
    }
  }
  return prefix + postfix;
}

void Broadcast::accept(IRVisitor *visitor) { visitor->visit(this); }

// Relu
Relu::Relu(const TypePtr &opType, const ValuePtr &inValue) : Node(opType) {
  this->inValue = inValue;
}
Relu::operator std::string() const {
  return getName() + " = ailang::relu(" + getValue()->getName() +
         "):" + std::string(*getType());
}
void Relu::accept(IRVisitor *visitor) { visitor->visit(this); }

std::vector<int> Relu::getShape() {
  if (auto tensorType = dynamic_cast<TensorType *>(inValue->getType().get())) {
    return tensorType->getConcreteShape();
  } else {
    throw std::runtime_error("ReLU input is not a tensor");
  }
}
// Mean

Mean::Mean(const TypePtr &opType, const ValuePtr &inValue,
           const std::vector<int64_t> &dim)
    : Node(opType), inValue(inValue), dim(dim) {}
Mean::operator std::string() const {
  auto prefix = getName() + " = ailang::mean(" + getValue()->getName() + ")";
  std::string postfix = "<dim=[";
  for (auto d : dim) {
    if (d == dim.back())
      postfix += std::to_string(d) + "]>";
    else
      postfix += std::to_string(d) + ",";
  }
  postfix += ":" + std::string(*getType());
  return prefix + postfix;
}

void Mean::accept(IRVisitor *visitor) { visitor->visit(this); }

std::vector<int> Mean::getShape() {
  if (auto tensorType = dynamic_cast<TensorType *>(inValue->getType().get())) {
    return tensorType->getConcreteShape();
  } else {
    throw std::runtime_error("Mean input is not a tensor");
  }
}
// Variance

Variance::Variance(const TypePtr &opType, const ValuePtr &inValue,
                   const std::vector<int64_t> &dim, const int ddof)
    : Node(opType), inValue(inValue), dim(dim),ddof(ddof) {}
Variance::operator std::string() const {
  auto prefix =
      getName() + " = ailang::variance(" + getValue()->getName() + ")";
  std::string postfix = "<dim=[";
  for (auto d : dim) {
    if (d == dim.back())
      postfix += std::to_string(d) + "]>";
    else
      postfix += std::to_string(d) + ",";
  }
  postfix += ":" + std::string(*getType());
  return prefix + postfix;
}

void Variance::accept(IRVisitor *visitor) { visitor->visit(this); }

std::vector<int> Variance::getShape() {
  if (auto tensorType = dynamic_cast<TensorType *>(inValue->getType().get())) {
    return tensorType->getConcreteShape();
  } else {
    throw std::runtime_error("Variance input is not a tensor");
  }
}
// Transpose
Transpose::Transpose(const TypePtr &opType, const ValuePtr &inValue)
    : Node(opType) {
  setUse(inValue, 0);
  this->inValue = inValue;
}
Transpose::operator std::string() const {
  return getName() + " = ailang::transpose(" + getValue()->getName() +
         "):" + std::string(*getType());
}

void Transpose::accept(IRVisitor *visitor) { visitor->visit(this); }

std::vector<int> Transpose::getShape() {
  if (auto tensorType = dynamic_cast<TensorType *>(inValue->getType().get())) {
    return tensorType->getConcreteShape();
  } else {
    throw std::runtime_error("Transpose input is not a tensor");
  }
}

// Maxpool2d
Maxpool2d::Maxpool2d(const TypePtr &opType, const ValuePtr &inValue,
                     const std::vector<int64_t> &window_dimensions,
                     const std::vector<int64_t> &window_strides,
                     const std::vector<int64_t> &base_dilations,
                     const std::vector<int64_t> &window_dilations,
                     const std::vector<int64_t> &padding)
    : Node(opType), window_dimensions(window_dimensions),
      window_strides(window_strides), base_dilations(base_dilations),
      window_dilations(window_dilations), padding(padding) {
  this->inValue = inValue;
}
Maxpool2d::operator std::string() const {
  return getName() + " = ailang::maxpool2d(" + getValue()->getName() +
         "):" + std::string(*getType());
}
void Maxpool2d::accept(IRVisitor *visitor) { visitor->visit(this); }

// Avgpool2d
Avgpool2d::Avgpool2d(const TypePtr &opType, const ValuePtr &inValue,
                     const std::vector<int64_t> &window_dimensions,
                     const std::vector<int64_t> &window_strides,
                     const std::vector<int64_t> &base_dilations,
                     const std::vector<int64_t> &window_dilations,
                     const std::vector<int64_t> &padding)
    : Node(opType), window_dimensions(window_dimensions),
      window_strides(window_strides), base_dilations(base_dilations),
      window_dilations(window_dilations), padding(padding) {
  this->inValue = inValue;
}
Avgpool2d::operator std::string() const {
  return getName() + " = ailang::avgpool2d(" + getValue()->getName() +
         "):" + std::string(*getType());
}
void Avgpool2d::accept(IRVisitor *visitor) { visitor->visit(this); }

// Convolution
Convolution::Convolution(const TypePtr &opType, const ValuePtr &inputValue,
                         const ValuePtr &weightValue,
                         const std::vector<int64_t> &window_strides,
                         const std::vector<int64_t> &lhsDilation,
                         const std::vector<int64_t> &rhsDilation,
                         const std::vector<int64_t> &padding_args,
                         const std::vector<int64_t> &window_reversal)
    : Node(opType), window_strides(window_strides), lhsDilation(lhsDilation),
      rhsDilation(rhsDilation), padding_args(padding_args),
      window_reversal(window_reversal) {
  this->inputValue = inputValue;
  this->weightValue = weightValue;
}
Convolution::operator std::string() const {
  return getName() + " = ailang::convolution(" + getInputValue()->getName() +
         "," + getWeightValue()->getName() + "):" + std::string(*getType());
}
void Convolution::accept(IRVisitor *visitor) { visitor->visit(this); }

// BatchNorm2d
BatchNorm2d::BatchNorm2d(const TypePtr &opType, const ValuePtr &inValue,
                         const ValuePtr &scale, const ValuePtr &offset,
                         const ValuePtr &mean, const ValuePtr &variance)
    : Node(opType) {
  this->inValue = inValue;
  this->scale = scale;
  this->offset = offset;
  this->mean = mean;
  this->variance = variance;
}
BatchNorm2d::operator std::string() const {
  return getName() + " = ailang::batchnorm2d(" + getValue()->getName() +
         "):" + std::string(*getType());
}
void BatchNorm2d::accept(IRVisitor *visitor) { visitor->visit(this); }

// whileOp
WhileOp::WhileOp(const TypePtr &nodeType, const ModulePtr &condGraph,
                 const ModulePtr &bodyGraph, const std::vector<ValuePtr> &args)
    : Node(nodeType), cond(condGraph), body(bodyGraph), inits(std::move(args)) {
  if (nodeType->isTupleType()) {
    auto types = asType<TupleType>(nodeType)->getTypes();
    for (const auto &type : types) {
      outs.push_back(Node::create(type));
    }
  } else {
    throw std::runtime_error("WhileOp output type must be a tuple type.");
  }
}

WhileOp::operator std::string() const {
  std::string result;
  std::string lhs;
  std::string indent = "\t\t";

  auto addIndent = [&indent](const std::string &str) {
    std::stringstream ss(str);
    std::string line;
    std::string result;
    while (std::getline(ss, line)) {
      result += indent + line + "\n";
    }
    return result;
  };

  lhs += "(";
  for (size_t i = 0; i < outs.size(); i++) {
    if (i == outs.size() - 1) {
      lhs += outs[i]->getName() + ") ";
    } else {
      lhs += outs[i]->getName() + ", ";
    }
  }
  result += lhs + " = ailang::while (";
  for (size_t i = 0; i < inits.size(); i++) {
    if (i == inits.size() - 1) {
      result += inits[i]->getName() + "): ";
    } else {
      result += inits[i]->getName() + ", ";
    }
  }
  result += getType()->getName();
  result += " {\n" + addIndent(std::string(*cond)) +
            addIndent(std::string(*body)) + "\n\t}";
  return result;
}

CompareOp::CompareOp(const TypePtr &nodeType, const ValuePtr &lhs,
                     const ValuePtr &rhs, CompareOp::CompareType op)
    : Node(nodeType) {
  setUse(lhs, 0);
  setUse(rhs, 1);
  this->lhs = lhs;
  this->rhs = rhs;
  this->op = op;
}

CompareOp::operator std::string() const {
  return getName() + " = ailang::" + compareOpString[static_cast<size_t>(op)] +
         "(" + lhs->getName() + ", " + rhs->getName() +
         "): " + std::string(*getType());
}

void CompareOp::accept(IRVisitor *visitor) { visitor->visit(this); }

Concat::Concat(const TypePtr &nodeType, const std::vector<ValuePtr> &inputs,
               int dim)
    : Node(nodeType), inputs(inputs), dim(dim) {
  for (size_t i = 0; i < inputs.size(); i++) {
    setUse(inputs[i], i);
  }
}

Concat::operator std::string() const {
  std::string result;
  std::string lhs;
  for (size_t i = 0; i < inputs.size(); i++) {
    if (i == inputs.size() - 1) {
      lhs += inputs[i]->getName();
    } else {
      lhs += inputs[i]->getName() + ", ";
    }
  }
  result += getName() + " = ailang::concat(" + lhs + ", " +
            std::to_string(dim) + "): " + std::string(*getType());
  return result;
}

void Concat::accept(IRVisitor *visitor) { visitor->visit(this); }

ConstantDef::ConstantDef(const TypePtr &nodeType, const ValuePtr &value)
    : Node(nodeType), value(value) {
  setUse(value, 0);
}

ConstantDef::operator std::string() const {
  return getName() + " = ailang::constant(" + value->getName() +
         "): " + std::string(*getType());
}

void ConstantDef::accept(IRVisitor *visitor) { visitor->visit(this); }

Exp::Exp(const TypePtr &nodeType, const ValuePtr &inValue)
    : Node(nodeType), inValue(inValue) {
  setUse(inValue, 0);
}

Exp::operator std::string() const {
  return getName() + " = ailang::exp(" + inValue->getName() +
         "): " + std::string(*getType());
}

void Exp::accept(IRVisitor *visitor) { visitor->visit(this); }

Tanh::Tanh(const TypePtr &nodeType, const ValuePtr &inValue)
    : Node(nodeType), inValue(inValue) {
  setUse(inValue, 0);
}

Tanh::operator std::string() const {
  return getName() + " = ailang::tanh(" + inValue->getName() +
         "): " + std::string(*getType());
}

void Tanh::accept(IRVisitor *visitor) { visitor->visit(this); }

Neg::Neg(const TypePtr &nodeType, const ValuePtr &inValue)
    : Node(nodeType), inValue(inValue) {
  setUse(inValue, 0);
}

Neg::operator std::string() const {
  return getName() + " = ailang::neg(" + inValue->getName() +
         "): " + std::string(*getType());
}

void Neg::accept(IRVisitor *visitor) { visitor->visit(this); }

Div::Div(const TypePtr &opType, const ValuePtr &lhs, const ValuePtr &rhs)
    : Node(opType) {
  setUse(lhs, 0);
  setUse(rhs, 1);
  this->lhs = lhs;
  this->rhs = rhs;
}

Div::operator std::string() const {
  return getName() + " = ailang::matmul(" + getOperand(0)->getName() + ", " +
         getOperand(1)->getName() + "): " + std::string(*getType());
}

void Div::accept(IRVisitor *visitor) { visitor->visit(this); }

IfOp::IfOp(const TypePtr &nodeType, const ModulePtr &trueBranch,
           const ModulePtr &falseBranch, const ValuePtr &cond)
    : Node(nodeType), trueBody(trueBranch), elseBody(falseBranch), cond(cond) {
  // setUse(cond, 0);
  if (nodeType->isTupleType()) {
    auto types = asType<TupleType>(nodeType)->getTypes();
    for (const auto &type : types) {
      auto *OutNode = Node::create(type);
      OutNode->setUse(cond, 0);
      outs.push_back(Node::create(type));
    }
  } else {
    auto *OutNode = Node::create(nodeType);
    OutNode->setUse(cond, 0);
    outs.push_back(Node::create(nodeType));
  }
}

IfOp::operator std::string() const {
  std::string result;
  std::string lhs;
  std::string indent = "\t\t";

  auto addIndent = [&indent](const std::string &str) {
    std::stringstream ss(str);
    std::string line;
    std::string result;
    while (std::getline(ss, line)) {
      result += indent + line + "\n";
    }
    return result;
  };

  for (size_t i = 0; i < outs.size(); i++) {
    if (i == outs.size() - 1) {
      lhs += outs[i]->getName();
    } else {
      lhs += outs[i]->getName() + ", ";
    }
  }
  result += lhs + " = ailang::if (";
  result += cond->getName() + ") : ";
  result += getType()->getName();
  result += " {\n" + addIndent(std::string(*trueBody)) + "\n\t} else {\n" +
            addIndent(std::string(*elseBody)) + "\n\t}";
  return result;
}

void IfOp::accept(IRVisitor *visitor) { visitor->visit(this); }

} // namespace ainl::ir