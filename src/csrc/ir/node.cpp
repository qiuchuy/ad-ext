#include <memory>

#include "block.h"
#include "function.h"
#include "node.h"

Node::Node() : Value() { init(); }

Node::Node(TypePtr type) : Value(type) {
    init();
    this->signature = new Signature(nullptr, type);
}

Node::Node(TypePtr type, const TypePtr &inType) : Value(type) {
    init();
    this->signature = new Signature(inType, type);
}

void Node::setUse(ValuePtr value, int idx) {
    auto use = new Use(this, value, idx);
    value->insertUseAtEnd(use);
    useList.push_back(use);
    useValueList.push_back(value);
}

void Node::addBlock() {
    BlockPtr newBlock = new Block();
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

void Node::addBlock(const std::vector<ValuePtr> &inValues) {
    auto newBlock = new Block(inValues);
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

Alloca::Alloca(const TypePtr &type)
    : Node(PointerType::createPointerType(type)) {
    this->contentType = type;
}
Load::Load(const ValuePtr &inVal)
    : Node((SAFE_TYPE_DOWNCAST(inVal->getType(), PointerType))
               ->getPointeeType()) {
    setUse(inVal, 0);
}

ValuePtr Load::getAddress() const { return useValueList[0]; }

Store::Store(const ValuePtr &inValue, const ValuePtr &address)
    : Node(VoidTypePtr::get()) {
    setUse(inValue, 0);
    setUse(address, 1);
}

ValuePtr Store::getValue() const { return useValueList[0]; }
ValuePtr Store::getAddress() const { return useValueList[1]; }

Param::Param() : Node(VoidTypePtr::get()) {}
Param::Param(const std::vector<ValuePtr> &params, const TypePtr &type)
    : Node(VoidTypePtr::get(), type) {
    this->params = params;
    this->contentType = type;
}

ReturnOp::ReturnOp() : Node(VoidTypePtr::get()) {}
ReturnOp::ReturnOp(const std::vector<ValuePtr> &params, const TypePtr &type)
    : Node(VoidTypePtr::get(), type) {
    this->params = params;
    this->contentType = type;
}
