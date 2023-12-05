#ifndef AINL_SRC_INCLUDE_AST_BINDING_H
#define AINL_SRC_INCLUDE_AST_BINDING_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include "ast.h"
#include "ast_node.h"
#include "logger.h"
#include "utils.h"

namespace py = pybind11;

class AstTransformer {
  public:
    AstTransformer() = default;

    static Module convertModule(const py::list &stmtList) {
        std::vector<Stmt> stmts;
        for (const auto &stmt : stmtList) {
            stmts.push_back(stmt.cast<Stmt>());
        }
        return std::make_shared<ModuleNode>(stmts);
    }

    static FunctionDef
    convertFunctionDef(const std::string &name,
                       const std::vector<std::string> &arguments,
                       const std::vector<Stmt> &stmtList) {
        return std::make_shared<FunctionDefNode>(name, arguments, stmtList);
    }

    static Bind convertBind(const std::vector<Expr> &target,
                            const Expr &source) {
        return std::make_shared<BindNode>(target, source);
    }

    static Var convertVar(const std::string &name) {
        return std::make_shared<VarNode>(name);
    }

    static VarDef convertVarDef(const std::string &name) {
        return std::make_shared<VarDefNode>(name);
    }

    static Constant convertConstant(const std::string &value) {
        return std::make_shared<ConstantNode>(value);
    }

    static Tuple convertTuple(const std::vector<Expr> &elems) {
        return std::make_shared<TupleNode>(elems);
    }

    static BinaryOp convertBinaryOp(const std::string &op, const Expr &op1,
                                    const Expr &op2) {
        return std::make_shared<BinaryOpNode>(BinaryOpASTHelper(op), op1, op2);
    }

    static UnaryOp convertUnaryOp(const std::string &op, const Expr &value) {
        return std::make_shared<UnaryOpNode>(UnaryOpASTHelper(op), value);
    }

    static Call
    convertCall(const Expr &func, const std::vector<Expr> &args,
                const std::unordered_map<std::string, Expr> &keywargs) {
        std::vector<Expr> callArgs(args);
        for (const auto &keywarg : keywargs) {
            callArgs.push_back(keywarg.second);
        }
        return std::make_shared<CallNode>(func, callArgs);
    }

    static Compare convertCompare(const Expr &left,
                                  const std::vector<std::string> &ops,
                                  const std::vector<Expr> &comparators) {
        std::vector<CompareNode::CompareOpKind> iops;
        for (const auto &op : ops)
            iops.push_back(CompareOpASTHelper(op));
        return std::make_shared<CompareNode>(left, iops, comparators);
    }

    static While convertWhile(const Expr &cond, std::vector<Stmt> &body) {
        return std::make_shared<WhileNode>(cond, body);
    }

    static Return convertReturn(const Expr &value) {
        return std::make_shared<ReturnNode>(value);
    }
};

void initAST(py::module_ &m);

#endif // AINL_SRC_INCLUDE_AST_BINDING_H