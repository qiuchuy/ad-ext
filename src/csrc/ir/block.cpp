#include "block.h"

Block::Block() {
    paramNode = Param::create();
    returnNode = Return::create();
    NodePtr beginNode = new Node();
    NodePtr endNode = new Node();
    beginNode->setNext(endNode);
    endNode->setPrev(beginNode);
    beginBlock = endBlock = nullptr;
}

Block::Block(const std::vector<ValuePtr> &inValues) {
    TypePtr inType = createTypePtrForValues(inValues);
    paramNode = Param::create(inValues, inType);
    returnNode = Return::create();
    NodePtr beginNode = new Node();
    NodePtr endNode = new Node();
    beginNode->setNext(endNode);
    endNode->setPrev(beginNode);
    beginBlock = endBlock = nullptr;
}

Block::Block(const std::vector<ValuePtr> &inValues,
             const std::vector<ValuePtr> &outValues) {
    TypePtr inType = createTypePtrForValues(inValues);
    TypePtr outType = createTypePtrForValues(outValues);
    paramNode = Param::create(inValues, inType);
    returnNode = Return::create(outValues, outType);
    NodePtr beginNode = new Node();
    NodePtr endNode = new Node();
    beginNode->setNext(endNode);
    endNode->setPrev(beginNode);
    beginBlock = endBlock = nullptr;
}

void Block::insertNodeAtEnd(NodePtr node) { endNode->insertBefore(node); }