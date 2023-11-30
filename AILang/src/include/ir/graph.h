#ifndef AINL_SRC_INCLUDE_GRAPH_H
#define AINL_SRC_INCLUDE_GRAPH_H

#include <list>
#include <memory>

#include "block.h"
#include "function.h"
#include "node.h"

class Node;
class Block;
class Signature;
class Graph;
using NodePtr = Node *;
using BlockPtr = Block *;
using SignaturePtr = Signature *;
using GraphPtr = std::shared_ptr<Graph>;

class Graph : public std::enable_shared_from_this<Graph>, public Value {
  public:
    class GraphParam : public Value {
      public:
        static int FPARAM_COUNT;
        int idx;
        explicit operator std::string() const override;
        // friend class Graph;
        // friend class ALModule;
      public:
        GraphParam(TypePtr type, int idx);
        ~GraphParam() override = default;
    };
    explicit Graph(std::string name);
    std::vector<ValuePtr> getParams() {
        return ((BlockPtr)(beginBlock->next))->getParams();
    }
    template <typename NodeType, typename... ARGS>
    NodePtr create(ARGS &&... args) {
        NodePtr Node = new NodeType(std::forward<ARGS>(args)...);
        Node->graph = shared_from_this();
        Node->block = (BlockPtr)(endBlock->prev);
        insertNodeAtEnd(Node);
        return Node;
    }
    std::string getName() const override;
    friend class Node;
    friend class ALModule;
    // friend std::ostream &operator<<(std::ostream &stream, const GraphPtr &g);
    std::string str();

  private:
    void insertNodeAtEnd(NodePtr Node);

    BlockPtr beginBlock;
    BlockPtr endBlock;
    std::string name;
};
using GraphPtr = std::shared_ptr<Graph>;

#endif // AINL_SRC_INCLUDE_GRAPH_H
