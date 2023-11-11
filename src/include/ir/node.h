#ifndef AINL_SRC_INCLUDE_NODE_H
#define AINL_SRC_INCLUDE_NODE_H

#include "block.h"
#include "function.h"
#include "graph.h"
#include "type.h"
#include "value.h"

#define NODE_PTR_TYPE_DECL(name) using name##Ptr = name *;

// Language builtin operators
// Extend this class to add more operators
class Graph;
class Block;
class Signature;
using GraphPtr = std::shared_ptr<Graph>;
using BlockPtr = Block *;
using SignaturePtr = Signature *;

class Node : public Value {
  public:
    enum class NodeKind {
        ALLOCA,
        STORE,
        LOAD,
        GETELEMENTPTR,
        ASSIGN,
        ADD,
        SUB,
        MUL,
        DIV,
        FLOORDIV,
        MOD,
        POW,
        LSHIFT,
        RSHIFT,
        BITOR,
        BITXOR,
        BITAND,
        MATMUL,
        CALL,
        PARAM,
        RETURN,
        MAKETUPLE,
        UNZIPPING,
        UNKNOWN,
    };

    void init() {
        useList.clear();
        useValueList.clear();
        graph = nullptr;
        signature = nullptr;
        block = nullptr;
    }
    ~Node() override = default;
    Node();
    explicit Node(TypePtr type);
    Node(TypePtr type, const TypePtr &inType);

    void addBlock();
    void addBlock(const std::vector<ValuePtr> &inValues);
    void addBlock(const std::vector<ValuePtr> &inValues,
                  const std::vector<ValuePtr> &outValues);

    std::string getName() const override { return str(); }
    explicit operator std::string() { return str(); }
    virtual NodeKind kind() { return Node::NodeKind::UNKNOWN; }

    friend class Graph;

  protected:
    void setUse(ValuePtr value, int idx);
    virtual std::string str() const { return ""; }

  protected:
    std::vector<ValuePtr> useValueList;
    std::vector<UsePtr> useList;
    SignaturePtr signature;
    GraphPtr graph;
    BlockPtr block;
};
NODE_PTR_TYPE_DECL(Node)

class Param : public Node {
  public:
    Param();
    Param(const std::vector<ValuePtr> &params, const TypePtr &types);
    static NodePtr create() { return new Param(); }
    static NodePtr create(const std::vector<ValuePtr> &params,
                          const TypePtr &types) {
        return new Param(params, types);
    }
    NodeKind kind() override { return Node::NodeKind::PARAM; }
    std::string str() const override {
        std::stringstream ssm;
        ssm << "(";
        if (!params.empty()) {
            for (size_t i = 0; i < params.size() - 1; i++)
                ssm << params[i]->getName() << " : "
                    << params[i]->getType()->str() << ", ";
            ssm << params[params.size() - 1]->getName() << " : "
                << params[params.size() - 1]->getType()->str();
        }
        ssm << ")";
        return ssm.str();
    }

  private:
    std::vector<ValuePtr> params;
    TypePtr contentType;
};
NODE_PTR_TYPE_DECL(Param)

class Return : public Node {
  public:
    Return();
    Return(const std::vector<ValuePtr> &params, const TypePtr &type);
    static NodePtr create() { return new Return(); }
    static NodePtr create(const std::vector<ValuePtr> &params,
                          const TypePtr &types) {
        return new Return(params, types);
    }

    NodeKind kind() override { return NodeKind::RETURN; }
    std::string str() const override {
        std::stringstream ssm;
        ssm << "return ";
        if (!params.empty()) {
            for (size_t i = 0; i < params.size() - 1; i++)
                ssm << params[i]->getName() << ", ";
            ssm << params[params.size() - 1]->getName();
        }
        ssm << ")";
        return ssm.str();
    }

  private:
    std::vector<ValuePtr> params;
    TypePtr contentType;
};
NODE_PTR_TYPE_DECL(Return)

class Alloca : public Node {
  public:
    explicit Alloca(const TypePtr &type);
    NodeKind kind() override { return Node::NodeKind::ALLOCA; }
    std::string str() const override {
        return getName() + " = alloca " + contentType->str();
    }

  private:
    TypePtr contentType;
};
NODE_PTR_TYPE_DECL(Alloca)

class Load : public Node {
  public:
    Load(const ValuePtr &inValue);
    ValuePtr getAddress() const;
    NodeKind kind() override { return Node::NodeKind::LOAD; }
    std::string str() const override {
        return getName() + " = load " + getAddress()->getName();
    }
};
NODE_PTR_TYPE_DECL(Load)

class Store : public Node {
  public:
    Store(const ValuePtr &inValue, const ValuePtr &address);
    ValuePtr getAddress() const;
    ValuePtr getValue() const;
    NodeKind kind() override { return Node::NodeKind::STORE; }
    std::string str() const override {
        return "store " + getValue()->getName() + ", " +
               getAddress()->getName();
    }
};
NODE_PTR_TYPE_DECL(Store)

#endif // AINL_SRC_INCLUDE_NODE_H