#ifndef AINL_SRC_INCLUDE_AST_H
#define AINL_SRC_INCLUDE_AST_H

#include <memory>
#include <utility>

#include "type.h"

class Visitor;

#define ASTNODE_TYPE(asttype)                                                  \
    virtual bool is##asttype##Node() { return false; }

#define SAFE_AST_DOWNCAST(shared_ptr, derived_type)                            \
    std::dynamic_pointer_cast<derived_type>(shared_ptr)

class ASTNode : public std::enable_shared_from_this<ASTNode> {
  public:
    enum class ASTNodeKind {
        Module = 0,
        Expr,
        Stmt,
        Var,
        VarDef,
        Bind,
        Constant,
        UnaryOp,
        BinaryOp,
        FunctionDef,
        Return,
        Compare,
        WhileLoop,
        If,
        Call,
        Tuple,
        //---
        NumNodes,
    };

    ~ASTNode() = default;

    ASTNODE_TYPE(Module)
    ASTNODE_TYPE(Expr)
    ASTNODE_TYPE(Stmt)
    ASTNODE_TYPE(Var)
    ASTNODE_TYPE(Bind)
    ASTNODE_TYPE(Constant)
    ASTNODE_TYPE(UnaryOp)
    ASTNODE_TYPE(BinaryOp)
    ASTNODE_TYPE(FunctionDef)
    ASTNODE_TYPE(Return)
    ASTNODE_TYPE(Compare)
    ASTNODE_TYPE(While)
    ASTNODE_TYPE(If)
    ASTNODE_TYPE(Call)
    ASTNODE_TYPE(Tuple)

    std::string getName() const { return str(); }
    bool match(const ASTNode &other) const {
        return this->hash() == other.hash();
    }
    virtual ASTNodeKind kind() const { return ASTNodeKind::NumNodes; }
    virtual std::string str() const { return ""; }
    virtual void accept(Visitor *visitor);
    bool operator==(const ASTNode &other) const {
        return this->hash() == other.hash();
    }
    bool operator!=(const ASTNode &other) const {
        return this->hash() != other.hash();
    }
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
    void setType(TypePtr inType) { this->type = std::move(inType); }
    ASTNodeKind kind() const override { return ASTNodeKind::Expr; }
    std::string str() const override { return ""; }
    void accept(Visitor *visitor) override;
    bool isExprNode() override { return true; }

  protected:
    TypePtr type;
};
using Expr = std::shared_ptr<ExprNode>;

class StmtNode : public ASTNode {
  public:
    StmtNode() = default;
    ASTNodeKind kind() const override { return ASTNodeKind::Stmt; }
    std::string str() const override { return ""; }
    void accept(Visitor *visitor) override;
    bool isStmtNode() override { return true; }
};
using Stmt = std::shared_ptr<StmtNode>;

#endif // AINL_SRC_INCLUDE_AST_H