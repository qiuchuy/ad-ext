#pragma once

#include <iterator>
#include <list>
#include <memory>

#include "ir/block.h"
#include "ir/function.h"
#include "ir/node.h"

namespace ainl::ir {

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

  struct GraphIterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = BlockPtr;
    using reference = const value_type;

    explicit GraphIterator(BlockPtr block, BlockPtr begin, BlockPtr end)
        : block(block), beginBlock(begin), endBlock(end) {}

    reference operator*() const { return block; }

    GraphIterator &operator+(difference_type diff) {
      for (difference_type i = 0; i < diff; ++i) {
        if (block == endBlock) // Stop if we reach endBlock
          break;
        block = (BlockPtr)block->next;
      }
      return *this;
    }

    GraphIterator &operator++() {
      if (block != endBlock)
        block = (BlockPtr)block->next;
      return *this;
    }

    GraphIterator operator++(int) {
      GraphIterator temp = *this;
      block = (BlockPtr)block->next;
      return temp;
    }

    friend bool operator==(const GraphIterator &a, const GraphIterator &b) {
      return a.block == b.block;
    };

    friend bool operator!=(const GraphIterator &a, const GraphIterator &b) {
      return !(a == b);
    };

  private:
    BlockPtr block;
    BlockPtr beginBlock;
    BlockPtr endBlock;
  };

  GraphIterator begin() const {
    return GraphIterator((BlockPtr)beginBlock->next, beginBlock, endBlock);
  }

  GraphIterator end() const {
    return GraphIterator(endBlock, beginBlock, endBlock);
  }

private:
  void insertNodeAtEnd(NodePtr Node);

  BlockPtr beginBlock;
  BlockPtr endBlock;
  std::string name;
};
using GraphPtr = std::shared_ptr<Graph>;
} // namespace ainl::ir
