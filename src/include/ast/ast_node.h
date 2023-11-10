#ifndef AINL_SRC_INCLUDE_AST_NODE_H
#define AINL_SRC_INCLUDE_AST_NODE_H

#include <sstream>
#include <utility>

#include "ast.h"
#include "logger.h"

class Visitor;

class ModuleNode : public StmtNode {
  public:
    ModuleNode() = default;
    explicit ModuleNode(std::vector<Stmt> stmts) : stmts(std::move(stmts)) {}
    ASTNodeKind kind() const override { return ASTNodeKind::Module; }
    std::string str() const override {
        std::stringstream ssm;
        ssm << "Module["
            << "\n";
        for (const auto &stmt : stmts) {
            ssm << "\t" << stmt->getName() << "\n";
        }
        ssm << "]"
            << "\n";
        return ssm.str();
    }
    void accept(Visitor *visitor) override;
    bool isModuleNode() override { return true; }

  private:
    size_t hash() const override {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        // Combine the hash of the node's fields
        seed ^= stringHash("Module") + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        // Recursively hash the child nodes (subtrees)
        for (const Stmt &stmt : stmts) {
            seed ^= stmt->hash();
        }
        return seed;
    }

  private:
    std::vector<Stmt> stmts;
};
using Module = std::shared_ptr<ModuleNode>;

class VarNode : public ExprNode {
  public:
    VarNode() = default;
    explicit VarNode(std::string name) : name(std::move(name)) {}
    VarNode(std::string name, TypePtr type)
        : ExprNode(std::move(type)), name(std::move(name)) {}
    ASTNodeKind kind() const override { return ASTNodeKind::Var; }
    std::string str() const override { return name; }
    std::string getName() { return name; }
    bool isVarNode() override { return true; }

  private:
    size_t hash() const override {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        // Combine the hash of the node's fields
        seed ^= stringHash(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
    void accept(Visitor *visitor) override;

  private:
    std::string name;
};
using Var = std::shared_ptr<VarNode>;

class VarDefNode : public StmtNode {
  public:
    VarDefNode() = default;
    VarDefNode(std::vector<Expr> targets, Expr source)
        : targets(std::move(targets)), source(std::move(source)) {}
    ASTNodeKind kind() const override { return ASTNodeKind::VarDef; }
    std::string str() const override {
        std::stringstream ssm;
        for (size_t i = 0; i < targets.size() - 1; i++)
            ssm << targets[i]->getName() << ", ";
        ssm << targets[targets.size() - 1]->getName();
        ssm << " = " << source->getName();
        return ssm.str();
    }
    std::vector<Expr> getTargets() { return targets; }
    Expr getSource() { return source; }
    void accept(Visitor *visitor) override;
    bool isVarDefNode() override { return true; }

  private:
    size_t hash() const override {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        // Combine the hash of the node's fields
        seed ^= stringHash("=") + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        // Recursively hash the child nodes (subtrees)
        for (const Expr &expr : targets) {
            seed ^= expr->hash();
        }
        seed ^= source->hash();
        return seed;
    }

  private:
    std::vector<Expr> targets;
    Expr source;
};
using VarDef = std::shared_ptr<VarDefNode>;

class TupleNode : public ExprNode {
  public:
    TupleNode() = default;
    explicit TupleNode(std::vector<Expr> elems) : elems(std::move(elems)) {}
    explicit TupleNode(std::vector<Expr> elems,
                       const std::vector<TypePtr> &type)
        : elems(std::move(elems)) {
        this->type = TupleType::createUnnamedTuple(type);
    }
    ASTNodeKind kind() const override { return ASTNodeKind::Constant; }
    std::string str() const override {
        std::stringstream ssm;
        ssm << "(";
        for (size_t i = 0; i < elems.size() - 1; i++)
            ssm << elems[i]->getName() << ", ";
        ssm << elems[elems.size() - 1]->getName() << ")";
        return ssm.str();
    }
    void accept(Visitor *visitor) override;
    bool isTupleNode() override { return true; }
    std::vector<Expr> getElems() { return elems; }

  private:
    size_t hash() const override {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        // Combine the hash of the node's fields
        seed ^= stringHash("()") + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        // Recursively hash the child nodes (subtrees)
        for (const Expr &elem : elems) {
            seed ^= elem->hash();
        }
        return seed;
    }

  private:
    std::vector<Expr> elems;
};
using Tuple = std::shared_ptr<TupleNode>;

class ConstantNode : public ExprNode {
  public:
    ConstantNode() = default;
    explicit ConstantNode(std::string value) : value(std::move(value)) {}
    explicit ConstantNode(std::string value, TypePtr type)
        : ExprNode(std::move(type)), value(std::move(value)) {}
    ASTNodeKind kind() const override { return ASTNodeKind::Constant; }
    std::string str() const override { return value; }
    std::string getValue() { return value; }
    void accept(Visitor *visitor) override;
    bool isConstantNode() override { return true; }

  private:
    size_t hash() const override {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        // Combine the hash of the node's fields
        seed ^= stringHash(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

  private:
    std::string value;
};
using Constant = std::shared_ptr<ConstantNode>;

extern std::array<std::string, 4> UnaryOpString;
class UnaryOpNode : public ExprNode {
  public:
    enum class UnaryOpKind {
        UAdd = 0,
        USub,
        Not,
        Invert,
        //---
        NumUnaryOp,
    };
    UnaryOpNode() = default;
    UnaryOpNode(UnaryOpKind op, Expr value) : op(op), value(std::move(value)) {}
    UnaryOpNode(UnaryOpKind op, Expr value, TypePtr type)
        : ExprNode(std::move(type)), op(op), value(std::move(value)) {}
    ASTNodeKind kind() const override { return ASTNodeKind::UnaryOp; }
    std::string str() const override {
        return UnaryOpString[(size_t)op] + " " + value->getName();
    }
    Expr getOperand() { return value; }
    bool isUnaryOpNode() override { return true; }
    void accept(Visitor *visitor) override;

  private:
    size_t hash() const override {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        // Combine the hash of the node's fields
        seed ^= stringHash(UnaryOpString[(size_t)op]) + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);
        // Recursively hash the child nodes (subtrees)
        seed ^= value->hash();
        return seed;
    }

  private:
    UnaryOpKind op;
    Expr value;
};
using UnaryOp = std::shared_ptr<UnaryOpNode>;
static_assert(UnaryOpString.size() ==
              (size_t)(UnaryOpNode::UnaryOpKind::NumUnaryOp));

extern std::array<std::string, 13> BinaryOpString;
class BinaryOpNode : public ExprNode {
  public:
    enum class BinaryOpKind {
        // Arith BinaryOp
        Add,
        Sub,
        Mul,
        Div,
        FloorDiv,
        Mod,
        Pow,
        LShift,
        RShift,
        BitOr,
        BitXor,
        BitAnd,
        MatMult,
        // Comparison BinaryOp
        //---,
        NumBinaryOp,
    };
    BinaryOpNode() = default;
    BinaryOpNode(BinaryOpKind op, Expr op1, Expr op2)
        : op(op), op1(std::move(op1)), op2(std::move(op2)) {}
    BinaryOpNode(BinaryOpKind op, Expr op1, Expr op2, TypePtr type)
        : ExprNode(std::move(type)), op(op), op1(std::move(op1)),
          op2(std::move(op2)) {}
    ASTNodeKind kind() const override { return ASTNodeKind::BinaryOp; }
    Expr getLHS() const { return op1; }
    Expr getRHS() const { return op2; }
    std::string str() const override {
        return op1->getName() + " " + BinaryOpString[(size_t)op] + " " +
               op2->getName();
    }
    void accept(Visitor *visitor) override;
    bool isBinaryOpNode() override { return true; }

  private:
    size_t hash() const override {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        // Combine the hash of the node's fields
        seed ^= stringHash(BinaryOpString[(size_t)op]) + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);
        // Recursively hash the child nodes (subtrees)
        seed ^= op1->hash();
        seed ^= op2->hash();
        return seed;
    }

  private:
    BinaryOpKind op;
    Expr op1;
    Expr op2;
};
using BinaryOp = std::shared_ptr<BinaryOpNode>;
static_assert(BinaryOpString.size() ==
              (size_t)BinaryOpNode::BinaryOpKind::NumBinaryOp);

class FunctionDefNode : public StmtNode {
  public:
    FunctionDefNode() = default;
    FunctionDefNode(std::string name, std::vector<std::string> params,
                    std::vector<Stmt> body)
        : name(std::move(name)), params(std::move(params)),
          body(std::move(body)) {}
    ASTNodeKind kind() const override { return ASTNodeKind::FunctionDef; }
    std::string str() const override {
        std::stringstream ssm;
        ssm << "func." << name << "(";
        for (size_t i = 0; i < params.size() - 1; i++)
            ssm << params[i] << ", ";
        ssm << params[params.size() - 1] << ")"
            << "[\n";
        for (const auto &stmt : body) {
            ssm << "\t" << stmt->getName() << "\n";
        }
        ssm << "]\n";
        return ssm.str();
    }
    void accept(Visitor *visitor) override;
    std::vector<std::string> getParams() const { return params; }
    std::string getName() { return name; }
    bool isFunctionDefNode() override { return true; }

  private:
    size_t hash() const override {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        // Combine the hash of the node's fields
        seed ^= stringHash(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        for (const std::string &param : params) {
            seed ^= stringHash(param);
        }
        // Recursively hash the child nodes (subtrees)
        for (const Stmt &stmt : body) {
            seed ^= stmt->hash();
        }
        return seed;
    }

  private:
    std::string name;
    std::vector<std::string> params;
    std::vector<Stmt> body;
};
using FunctionDef = std::shared_ptr<FunctionDefNode>;

class ReturnNode : public StmtNode {
  public:
    ReturnNode() = default;
    explicit ReturnNode(Expr value) : value(std::move(value)) {}
    ASTNodeKind kind() const override { return ASTNodeKind::Return; }
    std::string str() const override { return "return " + value->getName(); }
    bool isReturnNode() override { return true; }
    Expr getReturnValue() { return value; }
    void accept(Visitor *visitor) override;

  private:
    size_t hash() const override {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        // Combine the hash of the node's fields
        seed ^= stringHash("return") + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        // Recursively hash the child nodes (subtrees)
        seed ^= value->hash();
        return seed;
    }

  private:
    Expr value;
};
using Return = std::shared_ptr<ReturnNode>;

extern std::array<std::string, 10> CompareOpString;
class CompareNode : public ExprNode {
    // [TODO] This inheritance will cause a bug when the return value of Compare
    // Expression is not used or stored
  public:
    enum class CompareOpKind {
        Eq,
        NotEq,
        Lt,
        LtE,
        Gt,
        GtE,
        Is,
        IsNot,
        In,
        NotIn,
        // ---
        NumCompareOp,
    };
    CompareNode() = default;
    CompareNode(Expr left, std::vector<CompareOpKind> ops,
                std::vector<Expr> comparators)
        : ExprNode(BoolTypePtr::get()), left(std::move(left)),
          ops(std::move(ops)), comparators(std::move(comparators)) {}
    ASTNodeKind kind() const override { return ASTNodeKind::Compare; }
    std::string str() const override {
        std::stringstream ssm;
        ssm << left->getName() << " ";
        assert(ops.size() == comparators.size());
        for (size_t i = 0; i < ops.size(); i++)
            ssm << CompareOpString[(size_t)(ops[i])] << " "
                << comparators[i]->getName() << " ";
        return ssm.str();
    }
    void accept(Visitor *visitor) override;
    bool isCompareNode() override { return true; }

  private:
    size_t hash() const override {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        // Combine the hash of the node's fields
        seed ^= left->hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        // Recursively hash the child nodes (subtrees)
        for (size_t i = 0; i < ops.size(); i++) {
            seed ^= stringHash(CompareOpString[(size_t)(ops[i])]);
            seed ^= comparators[i]->hash();
        }
        return seed;
    }

  private:
    Expr left;
    std::vector<CompareOpKind> ops;
    std::vector<Expr> comparators;
};
using Compare = std::shared_ptr<CompareNode>;
static_assert(CompareOpString.size() ==
              (size_t)CompareNode::CompareOpKind::NumCompareOp);

class WhileNode : public StmtNode {
  public:
    WhileNode() = default;
    WhileNode(Expr cond, std::vector<Stmt> body)
        : cond(std::move(cond)), body(std::move(body)) {}
    ASTNodeKind kind() const override { return ASTNodeKind::WhileLoop; }
    std::string str() const override {
        std::stringstream ssm;
        ssm << "while(" << cond->getName() << ")[\n";
        for (const auto &stmt : body) {
            ssm << "\t" << stmt->getName() << "\n";
        }
        ssm << "]\n";
        return ssm.str();
    }
    void accept(Visitor *visitor) override;
    bool isWhileNode() override { return true; }

  private:
    size_t hash() const override {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        // Combine the hash of the node's fields
        seed ^= stringHash("while") + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= cond->hash();
        // Recursively hash the child nodes (subtrees)
        for (const Stmt &stmt : body) {
            seed ^= stmt->hash();
        }
        return seed;
    }

  private:
    Expr cond;
    std::vector<Stmt> body;
};
using While = std::shared_ptr<WhileNode>;

class IfNode : public StmtNode {
  public:
    IfNode() = default;
    ASTNodeKind kind() const override { return ASTNodeKind::If; }
    std::string str() const override {
        std::stringstream ssm;
        ssm << "if(" << cond->getName() << ")[\n";
        for (const auto &stmt : thenBranch) {
            ssm << "\t" << stmt->getName() << "\n";
        }
        ssm << "] else [\n";
        for (const auto &stmt : elseBranch) {
            ssm << "\t" << stmt->getName() << "\n";
        }
        ssm << "]\n";
        return ssm.str();
    }
    void accept(Visitor *visitor) override;
    bool isIfNode() override { return true; }

  private:
    size_t hash() const override {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        // Combine the hash of the node's fields
        seed ^= stringHash("if") + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= cond->hash();
        // Recursively hash the child nodes (subtrees)
        for (const Stmt &stmt : thenBranch) {
            seed ^= stmt->hash();
        }
        for (const Stmt &stmt : elseBranch) {
            seed ^= stmt->hash();
        }
        return seed;
    }

  private:
    Expr cond;
    std::vector<Stmt> thenBranch;
    std::vector<Stmt> elseBranch;
};
using If = std::shared_ptr<IfNode>;

class CallNode : public ExprNode {
  public:
    CallNode() = default;
    CallNode(Expr func, std::vector<Expr> args)
        : func(std::move(func)), args(std::move(args)) {}
    CallNode(Expr func, std::vector<Expr> args, TypePtr type)
        : ExprNode(std::move(type)), func(std::move(func)),
          args(std::move(args)) {}
    ASTNodeKind kind() const override { return ASTNodeKind::Call; }
    std::string str() const override {
        std::stringstream ssm;
        ssm << func->getName() << "(";
        for (size_t i = 0; i < args.size() - 1; i++)
            ssm << args[i] << ", ";
        ssm << args[args.size() - 1] << ")";
        return ssm.str();
    }
    void accept(Visitor *visitor) override;
    bool isCallNode() override { return true; }
    Expr getFunction() { return func; }

  private:
    size_t hash() const override {
        size_t seed = 0;
        std::hash<std::string> stringHash;
        // Combine the hash of the node's fields
        seed ^= stringHash("call") + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= func->hash();
        // Recursively hash the child nodes (subtrees)
        for (const Expr &arg : args) {
            seed ^= arg->hash();
        }
        return seed;
    }

  private:
    Expr func;
    std::vector<Expr> args;
};
using Call = std::shared_ptr<CallNode>;

#endif // AINL_SRC_INCLUDE_AST_NODE_H