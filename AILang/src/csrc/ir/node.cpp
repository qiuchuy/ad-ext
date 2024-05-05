#include "ir/node.h"

#include <memory>
#include <utility>

#include "ir/block.h"
#include "ir/function.h"
#include "ir/ir_visitor.h"

namespace ainl::ir {

int Node::LOCAL_COUNT = 0;

Node::Node() {
  init();
  this->signature = nullptr;
}

Node::Node(const TypePtr &type, const TypePtr &inType) : Value(type) {
  init();
  prefix = LOCAL_PREFIX;
  name = LOCAL_NAME_PREFIX + std::to_string(LOCAL_COUNT++);
  this->signature = new Signature(inType, type);
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

void Node::addBlockWithParam(NodePtr param) {
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
  this->graph->endBlock->insertBefore(newBlock);
}

void Node::accept(IRVisitor *visitor) { visitor->visit(this); }

Alloca::Alloca(const TypePtr &type)
    : Node(PointerType::createPointerType(type), VoidTypePtr::get()) {
  this->contentType = type;
}
Load::Load(const ValuePtr &inVal)
    : Node(
          (SAFE_TYPE_DOWNCAST(inVal->getType(), PointerType))->getPointeeType(),
          inVal->getType()) {
  setUse(inVal, 0);
}

ValuePtr Load::getAddress() const { return useValueList[0]; }

Store::Store(const ValuePtr &inValue, const ValuePtr &address)
    : Node(VoidTypePtr::get(), createTypePtrForValues({inValue, address})) {
  setUse(inValue, 0);
  setUse(address, 1);
}

ValuePtr Store::getValue() const { return useValueList[0]; }
ValuePtr Store::getAddress() const { return useValueList[1]; }

Param::Param(std::vector<ValuePtr> params, const TypePtr &type)
    : Node(VoidTypePtr::get(), type) {
  this->params = std::move(params);
  this->contentType = type;
}

ReturnOp::ReturnOp(const ValuePtr &value)
    : Node(VoidTypePtr::get(), value->getType()) {
  this->value = value;
}

void ReturnOp::accept(IRVisitor *visitor) { visitor->visit(this); }

// Matmul
Matmul::Matmul(const TypePtr &opType, const ValuePtr &lhs, const ValuePtr &rhs)
    : Node(opType, createTypePtrForValues({lhs, rhs})) {
  this->lhs = lhs;
  this->rhs = rhs;
}

Matmul::operator std::string() const {
  return getName() + " = ailang::matmul(" + getLHS()->getName() + ", " +
         getRHS()->getName() + "): " + std::string(*signature);
}

void Matmul::accept(IRVisitor *visitor) { visitor->visit(this); }

// Relu
Relu::Relu(const TypePtr &opType, const ValuePtr &inValue)
    : Node(opType, createTypePtrForValues({inValue})) {
  this->inValue = inValue;
}
Relu::operator std::string() const {
  return getName() + " = ailang::relu(" + getValue()->getName() +
         "):" + std::string(*signature);
}

// Transpose
Transpose::Transpose(const TypePtr &opType, const ValuePtr &inValue)
    : Node(opType, createTypePtrForValues({inValue})) {
  this->inValue = inValue;
}
Transpose::operator std::string() const {
  return getName() + " = ailang::transpose(" + getValue()->getName() +
         "):" + std::string(*signature);
}

void Transpose::accept(IRVisitor *visitor) { visitor->visit(this); }

// Maxpool2d
Maxpool2d::Maxpool2d(const TypePtr &opType, const ValuePtr &inValue)
    : Node(opType, createTypePtrForValues({inValue})) {
  this->inValue = inValue;
}
Maxpool2d::operator std::string() const {
  return getName() + " = ailang::maxpool2d(" + getValue()->getName() +
         "):" + std::string(*signature);
}
// Convolution
Convolution::Convolution(const TypePtr &opType, const ValuePtr &inValue)
    : Node(opType, createTypePtrForValues({inValue})) {
  this->inValue = inValue;
}
Convolution::operator std::string() const {
  return getName() + " = ailang::convolution(" + getValue()->getName() +
         "):" + std::string(*signature);
}
// BatchNorm2d
BatchNorm2d::BatchNorm2d(const TypePtr &opType, const ValuePtr &inValue)
    : Node(opType, createTypePtrForValues({inValue})) {
  this->inValue = inValue;
}
BatchNorm2d::operator std::string() const {
  return getName() + " = ailang::batchnorm2d(" + getValue()->getName() +
         "):" + std::string(*signature);
}
} // namespace ainl::ir