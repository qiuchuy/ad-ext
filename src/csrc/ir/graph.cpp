#include "graph.h"
#include "value.h"

template <typename NodeType, typename... ARGS>
NodePtr Graph::create(ARGS &&...args) {
    NodePtr node = new NodeType(std::forward<ARGS>(args)...);
    node->graph = shared_from_this();
    node->block = endBlock;
    insertNodeAtEnd(node);
    return node;
}

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