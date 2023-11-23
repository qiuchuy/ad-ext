#ifndef AINL_SRC_INCLUDE_BLOCK_H
#define AINL_SRC_INCLUDE_BLOCK_H

#include "node.h"
#include "value.h"

class Node;
using NodePtr = Node *;
class Param;
using ParamPtr = Param *;
class ReturnOp;
using ReturnOpPtr = ReturnOp *;

class Block;
using BlockPtr = Block *;
class Block : public Value {
  public:
    Block();
    explicit Block(const std::vector<ValuePtr> &inValues);
    std::vector<ValuePtr> getParams();
    void insertNodeAtEnd(NodePtr Node);
    friend class Node;
    friend class Graph;

  private:
    // Node link list
    NodePtr beginNode;
    NodePtr endNode;

    // created for each block
    ParamPtr paramNode;
    ReturnOpPtr returnNode;

    // nested scope
    BlockPtr beginBlock;
    BlockPtr endBlock;
};

#endif // AINL_SRC_INCLUDE_BLOCK_H
