#ifndef AINL_SRC_INCLUDE_AST_H
#define AINL_SRC_INCLUDE_AST_H

#include <memory>
#include <utility>

#include "type.h"

class ASTNode : public std::enable_shared_from_this<ASTNode> {
  public:
    enum class ASTNodeKind {
        Module = 0,
        Expr,
        Stmt,
        Var,
        VarDef,
        Constant,
        UnaryOp,
        BinaryOp,
        FunctionDef,
        Return,
        WhileLoop,
        If,
        Call,
        Tuple,
        //---
        NumNodes,
    };

    ~ASTNode() = default;
    std::string getName() const { return str(); }
    bool match(const ASTNode& other) const {return this->hash() == other.hash();}
    virtual ASTNodeKind kind() const { return ASTNodeKind::NumNodes; }
    virtual std::string str() const { return ""; }
    bool operator==(const ASTNode& other) const {return this->hash() == other.hash();}
    bool operator!=(const ASTNode& other) const {return this->hash() != other.hash();}

    virtual size_t hash() const {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        seed ^= stringHash(getName()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

using AST = std::shared_ptr<ASTNode>;

class ExprNode : public ASTNode {
  public:
    ExprNode() = default;
    explicit ExprNode(TypePtr type) : type(std::move(type)) {}
    TypePtr getType() { return type; }
    ASTNodeKind kind() const override { return ASTNodeKind::Expr; }
    std::string str() const override { return ""; }

  protected:
    TypePtr type;
};
using Expr = std::shared_ptr<ExprNode>;

class StmtNode : public ASTNode {
  public:
    StmtNode() = default;
    ASTNodeKind kind() const override { return ASTNodeKind::Stmt; }
    std::string str() const override { return ""; }
};
using Stmt = std::shared_ptr<StmtNode>;

#endif // AINL_SRC_INCLUDE_AST_H