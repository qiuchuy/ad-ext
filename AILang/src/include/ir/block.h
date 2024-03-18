#pragma once

#include "ir/node.h"
#include "ir/value.h"

namespace ainl::ir {
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
    Block(int idx);
    std::vector<ValuePtr> getParams();
    void insertNodeAtHead(NodePtr Node);
    void insertNodeAtEnd(NodePtr Node);
    friend class Node;
    friend class Graph;
    explicit operator std::string() const override;
    static int blockCount;

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

    // label
    std::string label;
};
} // namespace ainl::ir
