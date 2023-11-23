#ifndef AINL_SRC_INCLUDE_Node_H
#define AINL_SRC_INCLUDE_Node_H

#include "block.h"
#include "type.h"
#include "value.h"

#define NODE_PTR_TYPE_DECL(name)                                               \
    class name;                                                                \
    using name##Ptr = name *;

// Language builtin Nodes
// Extend this class to add more Nodes
class Graph;
class Block;
class Signature;
using GraphPtr = std::shared_ptr<Graph>;
using BlockPtr = Block *;
using SignaturePtr = Signature *;

NODE_PTR_TYPE_DECL(Node)
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
        signature = nullptr;
    }
    ~Node() override = default;
    Node();
    explicit Node(TypePtr type);
    Node(TypePtr type, const TypePtr &inType);

    std::string getName() const override { return "ailang::" + str(); }
    explicit operator std::string() { return str(); }
    virtual NodeKind kind() { return Node::NodeKind::UNKNOWN; }

    friend class Graph;

  protected:
    void addBlock();
    void addBlock(const std::vector<ValuePtr> &inValues);
    void setUse(ValuePtr value, int idx);
    virtual std::string str() const { return ""; }

  protected:
    std::vector<ValuePtr> useValueList;
    std::vector<UsePtr> useList;

    // They are actually not created by the constructor
    // Created by Graph
    SignaturePtr signature;
    // GraphPtr graph;
    // Created when creating a new local scope
    // BlockPtr block;
};

NODE_PTR_TYPE_DECL(Param)
class Param : public Node {
  public:
    Param();
    Param(const std::vector<ValuePtr> &params, const TypePtr &types);
    static ParamPtr create() { return new Param(); }
    static ParamPtr create(const std::vector<ValuePtr> &params,
                           const TypePtr &types) {
        return new Param(params, types);
    }
    std::vector<ValuePtr> getParams() { return params; }
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

NODE_PTR_TYPE_DECL(ReturnOp)
class ReturnOp : public Node {
  public:
    ReturnOp();
    ReturnOp(const std::vector<ValuePtr> &params, const TypePtr &type);
    static ReturnOpPtr create() { return new ReturnOp(); }
    static ReturnOpPtr create(const std::vector<ValuePtr> &params,
                              const TypePtr &types) {
        return new ReturnOp(params, types);
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

NODE_PTR_TYPE_DECL(Alloca)
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

NODE_PTR_TYPE_DECL(Load)
class Load : public Node {
  public:
    Load(const ValuePtr &inValue);
    ValuePtr getAddress() const;
    NodeKind kind() override { return Node::NodeKind::LOAD; }
    std::string str() const override {
        return getName() + " = load " + getAddress()->getName();
    }
};

NODE_PTR_TYPE_DECL(Store)
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

NODE_PTR_TYPE_DECL(Matmul)
class MatMul : public Node {
  public:
    MatMul(const ValuePtr &lhs, const ValuePtr &rhs);
    NodeKind kind() override { return Node::NodeKind::MATMUL; }
    std::string str() const override {
        return "matmul(" + getLHS()->getName() + ", " + getRHS()->getName() +
               ")";
    }
    ValuePtr getLHS() const { return lhs; }
    ValuePtr getRHS() const { return rhs; }

  private:
    ValuePtr lhs;
    ValuePtr rhs;
};

#endif // AINL_SRC_INCLUDE_Node_H