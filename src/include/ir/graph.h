#ifndef AINL_SRC_INCLUDE_GRAPH_H
#define AINL_SRC_INCLUDE_GRAPH_H

#include <list>
#include <memory>

#include "block.h"
#include "function.h"
#include "node.h"
#include "value.h"

class Node;
class Block;
class Signature;
using NodePtr = Node *;
using BlockPtr = Block *;
using SignaturePtr = Signature *;
class Graph : public std::enable_shared_from_this<Graph>, public Value {
  public:
    Graph(const std::vector<ValuePtr> &inValues,
          const std::vector<ValuePtr> &returnValues);

    template <typename NodeType, typename... ARGS>
    NodePtr create(ARGS &&...args);

    friend class Node;

  private:
    void insertNodeAtEnd(NodePtr node);

    SignaturePtr signature;
    BlockPtr beginBlock;
    BlockPtr endBlock;
};
using GraphPtr = std::shared_ptr<Graph>;

#endif // AINL_SRC_INCLUDE_GRAPH_H
