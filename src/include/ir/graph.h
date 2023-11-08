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
using NodePtr = Node *;
using BlockPtr = Block *;
class Graph : public std::enable_shared_from_this<Graph>, public Value {
  public:
    Graph(const std::vector<ValuePtr> &inValues,
          const std::vector<ValuePtr> &returnType);

    template <typename NodeType, typename... ARGS>
    NodePtr create(ARGS &&...args) {
        NodePtr node = new NodeType(std::forward<ARGS>(args)...);
        node->graph = shared_from_this();
        node->block = endBlock;
        insertNodeAtEnd(node);
        return node;
    }

    friend class Node;

  private:
    void insertNodeAtEnd(NodePtr node);

    SignaturePtr signature;
    BlockPtr beginBlock;
    BlockPtr endBlock;
};
using GraphPtr = std::shared_ptr<Graph>;

#endif // AINL_SRC_INCLUDE_GRAPH_H
