#include "block.h"

Block::Block() {
    paramNode = Param::create();
    returnNode = ReturnOp::create();
    beginNode = new Node();
    endNode = new Node();
    beginNode->setNext(endNode);
    endNode->setPrev(beginNode);
    beginBlock = endBlock = nullptr;
}

Block::Block(const std::vector<ValuePtr> &inValues) {

    for (auto &param : inValues) {
        param->block = this;
    }

    TypePtr inType = createTypePtrForValues(inValues);
    paramNode = Param::create(inValues, inType);
    returnNode = ReturnOp::create();
    beginNode = new Node();
    endNode = new Node();
    beginNode->setNext(endNode);
    endNode->setPrev(beginNode);
    beginBlock = endBlock = nullptr;
}

std::vector<ValuePtr> Block::getParams() { return paramNode->getParams(); }
void Block::insertNodeAtEnd(NodePtr Node) { endNode->insertBefore(Node); }