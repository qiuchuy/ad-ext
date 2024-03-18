#pragma once

#include <utility>

#include "ir/block.h"
#include "ir/type.h"
#include "ir/value.h"

namespace ainl::ir {


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
        RELU,
        TRANSPOSE,
        MAXPOOL2D,
        CONVOLUTION,
        BATCHNORM2d,
        UNKNOWN,
    };

    void init() {
        useList.clear();
        useValueList.clear();
        signature = nullptr;
    }
    ~Node() override = default;
    Node();
    Node(const TypePtr &type, const TypePtr &inType);

    explicit operator std::string() const override { return ""; }
    virtual NodeKind kind() { return Node::NodeKind::UNKNOWN; }

    friend class Graph;

    static int LOCAL_COUNT;

  public:
    void addBlock();
    void addBlockWithParam(NodePtr param);
    void setUse(ValuePtr value, int idx);

  protected:
    std::vector<ValuePtr> useValueList;
    std::vector<UsePtr> useList;

    // They are actually not created by the constructor
    // Created by Graph
    SignaturePtr signature{};
    // GraphPtr graph;
    // Created when creating a new local scope
    // BlockPtr block;
};

NODE_PTR_TYPE_DECL(Param)
class Param : public Node {
  public:
    Param(std::vector<ValuePtr> params, const TypePtr &types);
    static ParamPtr create(std::vector<ValuePtr> params, const TypePtr &types) {
        return new Param(std::move(params), types);
    }
    std::vector<ValuePtr> getParams() { return params; }
    NodeKind kind() override { return Node::NodeKind::PARAM; }
    explicit operator std::string() const override {
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
    explicit ReturnOp(const ValuePtr &value);
    static ReturnOpPtr create(ValuePtr value) { return new ReturnOp(value); }

    NodeKind kind() override { return NodeKind::RETURN; }
    explicit operator std::string() const override {
        std::stringstream ssm;
        ssm << "return ";
        ssm << value->getName();
        return ssm.str();
    }

  private:
    ValuePtr value;
};

NODE_PTR_TYPE_DECL(Alloca)
class Alloca : public Node {
  public:
    explicit Alloca(const TypePtr &type);
    NodeKind kind() override { return Node::NodeKind::ALLOCA; }
    explicit operator std::string() const override {
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
    explicit operator std::string() const override {
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
    explicit operator std::string() const override {
        return "store " + getValue()->getName() + ", " +
               getAddress()->getName();
    }
};

NODE_PTR_TYPE_DECL(Matmul)
class Matmul : public Node {
  public:
    Matmul(const TypePtr &nodeType, const ValuePtr &lhs, const ValuePtr &rhs);
    NodeKind kind() override { return Node::NodeKind::MATMUL; }
    explicit operator std::string() const override;
    ValuePtr getLHS() const { return lhs; }
    ValuePtr getRHS() const { return rhs; }

  private:
    ValuePtr lhs;
    ValuePtr rhs;
};

NODE_PTR_TYPE_DECL(Relu)
class Relu : public Node {
  public:
      
    Relu(const TypePtr &nodeType, const ValuePtr &inValue);
    NodeKind kind() override { return Node::NodeKind::RELU; }
    explicit operator std::string() const override;
    // 在需要将 Relu 类的对象转换为字符串类型时使用。
    ValuePtr getValue() const { return inValue; }

  private:
    ValuePtr inValue;
};

NODE_PTR_TYPE_DECL(Transpose)
class Transpose : public Node {
  public:
    // 似乎有同名类
    Transpose(const TypePtr &nodeType, const ValuePtr &inValue);
    NodeKind kind() override { return Node::NodeKind::TRANSPOSE; }
    explicit operator std::string() const override;
    // 在需要将 Transpsoe 类的对象转换为字符串类型时使用。
    ValuePtr getValue() const { return inValue; }

  private:
    ValuePtr inValue;
};

NODE_PTR_TYPE_DECL(Maxpool2d)
class Maxpool2d : public Node {
  public:
    Maxpool2d(const TypePtr &nodeType, const ValuePtr &inValue);
    NodeKind kind() override { return Node::NodeKind::MAXPOOL2D; }
    explicit operator std::string() const override;
    ValuePtr getValue() const { return inValue; }

  private:
    ValuePtr inValue;
};

NODE_PTR_TYPE_DECL(Convolution)
class Convolution : public Node {
  public:
    Convolution(const TypePtr &nodeType, const ValuePtr &inValue);
    NodeKind kind() override { return Node::NodeKind::CONVOLUTION; }
    explicit operator std::string() const override;
    ValuePtr getValue() const { return inValue; }

  private:
    ValuePtr inValue;
};

NODE_PTR_TYPE_DECL(BatchNorm2d)
class BatchNorm2d : public Node {
  public:
    BatchNorm2d(const TypePtr &nodeType, const ValuePtr &inValue);
    NodeKind kind() override { return Node::NodeKind::BATCHNORM2d; }
    explicit operator std::string() const override;
    ValuePtr getValue() const { return inValue; }

  private:
    ValuePtr inValue;
};
} // namespace ainl::ir