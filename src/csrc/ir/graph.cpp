#include "graph.h"
#include "value.h"

Graph::Graph(const std::vector<ValuePtr> &inputValues,
             const std::vector<ValuePtr> &returnValues) {
    TypePtr inType = createTypePtrForValues(inputValues);
    TypePtr outType = createTypePtrForValues(returnValues);
    this->signature = new Signature(inType, outType);
    this->beginBlock = new Block(inputValues);
    this->endBlock = new Block(returnValues);
    this->beginBlock->setNext(endBlock);
    this->endBlock->setPrev(beginBlock);
}

void Graph::insertNodeAtEnd(NodePtr node) {
    this->endBlock->insertNodeAtEnd(node);
}