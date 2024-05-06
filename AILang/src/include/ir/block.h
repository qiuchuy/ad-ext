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

  struct BlockIterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = NodePtr;
    using reference = const value_type;

    explicit BlockIterator(NodePtr node, NodePtr paramNode, NodePtr returnNode,
                           NodePtr beginNode, NodePtr endNode);
    BlockIterator &operator++();
    BlockIterator operator++(int);
    bool operator==(const BlockIterator &rhs) const;
    bool operator!=(const BlockIterator &rhs) const;
    reference operator*();

  private:
    NodePtr node;
    NodePtr beginNode;
    NodePtr endNode;
    NodePtr paramNode;
    NodePtr returnNode;
  };

  BlockIterator begin();
  BlockIterator end();

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
