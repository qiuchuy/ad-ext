#include "node.h"

#include <memory>

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
    UsePtr use = new Use(this, value, idx);
    value->insertUseAtEnd(use);
    useList.push_back(use);
    useValueList.push_back(value);
}

void Node::addBlock() {
    BlockPtr block = new Block();
    if (this->block->endBlock) {
        this->block->endBlock->insertBefore(block);
        return;
    }
    this->block->beginBlock = new Block();
    this->block->endBlock = new Block();
    this->block->beginBlock->setNext(this->block->endBlock);
    this->block->endBlock->setPrev(this->block->beginBlock);
    this->block->endBlock->insertBefore(block);
}

void Node::addBlock(const std::vector<ValuePtr> &inValues) {
    BlockPtr block = new Block(inValues);
    if (this->block->endBlock) {
        this->block->endBlock->insertBefore(block);
        return;
    }
    this->block->beginBlock = new Block();
    this->block->endBlock = new Block();
    this->block->beginBlock->setNext(this->block->endBlock);
    this->block->endBlock->setPrev(this->block->beginBlock);
    this->block->endBlock->insertBefore(block);
}

void Node::addBlock(const std::vector<ValuePtr> &inValues,
                    const std::vector<ValuePtr> &outValues) {
    BlockPtr block = new Block(inValues, outValues);
    if (this->block->endBlock) {
        this->block->endBlock->insertBefore(block);
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

Return::Return() : Node(VoidTypePtr::get()) {}
Return::Return(const std::vector<ValuePtr> &params, const TypePtr &type)
    : Node(VoidTypePtr::get(), type) {
    this->params = params;
    this->contentType = type;
}
