#ifndef AINL_SRC_INCLUDE_BLOCK_H
#define AINL_SRC_INCLUDE_BLOCK_H

#include "node.h"
#include "value.h"

class Block;
class Node;
using BlockPtr = Block *;
using NodePtr = Node *;
class Block : public Value {
  public:
    Block();
    explicit Block(const std::vector<ValuePtr> &inValues);
    Block(const std::vector<ValuePtr> &inValues,
          const std::vector<ValuePtr> &outValues);
    void insertNodeAtEnd(NodePtr node);
    friend class Node;

  private:
    // node link list
    NodePtr beginNode;
    NodePtr endNode;

    // created for each block
    NodePtr paramNode;
    NodePtr returnNode;

    // nested scope
    BlockPtr beginBlock;
    BlockPtr endBlock;
};

#endif // AINL_SRC_INCLUDE_BLOCK_H
