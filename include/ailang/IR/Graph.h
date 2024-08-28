#pragma once

#include <iterator>
#include <list>
#include <memory>

#include "ailang/IR/Block.h"
#include "ailang/IR/Node.h"

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

  public:
    GraphParam(TypePtr type, int idx);
    ~GraphParam() override = default;
  };
  explicit Graph();

  static std::shared_ptr<Graph> create(const TypePtr &inputType,
                                       const TypePtr &returnType) {
    auto graph = std::make_shared<Graph>();
    std::vector<ValuePtr> params;
    if (inputType->isTupleType()) {
      std::vector<TypePtr> paramTypes =
          asType<TupleType>(inputType)->getTypes();
      for (int idx = 0; (size_t)idx < paramTypes.size(); idx++) {
        auto param = new Graph::GraphParam(paramTypes[idx], idx);
        params.push_back(param);
      }
    } else {
      params.push_back(new Graph::GraphParam(inputType, 0));
    }
    for (auto &param : params) {
      param->graph = graph;
    }
    ParamPtr paramNode = Param::create(params, inputType);
    paramNode->graph = graph;
    paramNode->addBlockWithParam(paramNode, graph);
    return graph;
  }

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

  template <typename NodeType, typename... ARGS>
  NodePtr createAfter(NodePtr after, ARGS &&... args) {
    NodePtr Node = new NodeType(std::forward<ARGS>(args)...);
    Node->graph = shared_from_this();
    Node->block = after->block;
    insertNodeAfter(after, Node);
    return Node;
  }
  Value::ValueKind getValueKind() const override;
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
  void insertNodeAfter(NodePtr After, NodePtr Node);
  BlockPtr beginBlock;
  BlockPtr endBlock;
};
using GraphPtr = std::shared_ptr<Graph>;
} // namespace ainl::ir
