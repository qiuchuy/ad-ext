#include <memory>
#include <utility>

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
Param::Param(std::vector<ValuePtr> params, const TypePtr &type)
    : Node(VoidTypePtr::get(), type) {
    this->params = std::move(params);
    this->contentType = type;
}

ReturnOp::ReturnOp() : Node(VoidTypePtr::get()) { value = nullptr; }
ReturnOp::ReturnOp(const ValuePtr &value)
    : Node(VoidTypePtr::get(), value->getType()) {
    this->value = value;
}
