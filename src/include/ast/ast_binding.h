#ifndef AINL_SRC_INCLUDE_AST_BINDING_H
#define AINL_SRC_INCLUDE_AST_BINDING_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "ast.h"
#include "ast_node.h"
#include "logger.h"

namespace py = pybind11;

#define DISPATCH_BINARYOP(op)                                                  \
    if (OpKind == #op)                                                         \
        return BinaryOpNode::BinaryOpKind::op;
#define DISPATCH_UNARYOP(op)                                                   \
    if (OpKind == #op)                                                         \
        return UnaryOpNode::UnaryOpKind::op;

class AstTransformer {
  public:
    AstTransformer() = default;

    static BinaryOpNode::BinaryOpKind BinaryOpASTHelper(std::string OpKind) {
        DISPATCH_BINARYOP(Add)
        DISPATCH_BINARYOP(Sub)
        DISPATCH_BINARYOP(Mul)
        DISPATCH_BINARYOP(Div)
        DISPATCH_BINARYOP(FloorDiv)
        DISPATCH_BINARYOP(Mod)
        DISPATCH_BINARYOP(Pow)
        DISPATCH_BINARYOP(LShift)
        DISPATCH_BINARYOP(RShift)
        DISPATCH_BINARYOP(BitOr)
        DISPATCH_BINARYOP(BitXor)
        DISPATCH_BINARYOP(BitAnd)
        DISPATCH_BINARYOP(MatMult)
        throw AINLError("Unsupported BinaryOpNode when transforming AST.");
    }

    static UnaryOpNode::UnaryOpKind UnaryOpASTHelper(std::string OpKind) {
        DISPATCH_UNARYOP(UAdd)
        DISPATCH_UNARYOP(USub)
        DISPATCH_UNARYOP(Not)
        DISPATCH_UNARYOP(Invert)
        throw AINLError("Unsupported BinaryOpNode when transforming AST.");
    }

    static Module convertModule(const py::list &stmtList) {
        std::vector<Stmt> stmts;
        for (const auto &stmt : stmtList) {
            stmts.push_back(stmt.cast<Stmt>());
        }
        return std::make_shared<ModuleNode>(stmts);
    }

    static FunctionDef convertFunctionDef(const std::string &name,
                                          const std::vector<std::string> &arguments,
                                          const std::vector<Stmt> &stmtList) {
        return std::make_shared<FunctionDefNode>(name, arguments, stmtList);
    }

    static VarDef convertVarDef(const std::vector<Expr>& target, const Expr& source) {
      return std::make_shared<VarDefNode>(target, source);
    }

    static Var convertVar(const std::string& name) {
        return std::make_shared<VarNode>(name);
    }

    static Constant convertConstant(const std::string& value) {
        return std::make_shared<ConstantNode>(value);
    }

    static Tuple convertTuple(const std::vector<Expr>& elems) {
        return std::make_shared<TupleNode>(elems);
    }
};
/*
static std::shared_ptr<ModuleNode> convertModule(const py::list &stmtList) {
  std::vector<std::shared_ptr<StmtNode>> stmts;
  for (auto stmt : stmtList) {
    auto stmtNode = stmt.cast<std::shared_ptr<StmtNode>>();
    stmts.push_back(stmtNode);
  }
  return std::make_shared<ModuleNode>(stmts);
}

static std::shared_ptr<AssignNode> convertAssign(const py::list &targetList,
                                                 const py::object &source,
                                                 const py::object &init) {
  std::shared_ptr<ExprNode> targets;
  /// [TODO] yuqiuchu: extend TupleNode to enable tuple assignment
  bool isInit = init.cast<bool>();
  if (isInit) {
    for (auto target : targetList) {
      auto targetNode = target.cast<std::shared_ptr<LValInitNode>>();
      /// [TODO] yuqiuchu: use TupleNode
      /// targets.push_back(targetNnode);
      targets = targetNode;
    }
  } else {
    for (auto target : targetList) {
      auto targetNode = target.cast<std::shared_ptr<LValNode>>();
      /// [TODO] yuqiuchu: use TupleNode
      /// targets.push_back(targetNnode);
      targets = targetNode;
    }
  }
  auto src = source.cast<std::shared_ptr<ExprNode>>();
  auto assign = std::make_shared<AssignNode>(targets, src);
  return assign;
}
static std::shared_ptr<ExprNode> convertExpr(py::object Expr) {}
static std::shared_ptr<BinaryOpNode> convertBinaryExpr(py::object opname, const
py::object &lhs, const py::object &rhs) { std::string name =
opname.cast<std::string>(); auto lhs_expr =
lhs.cast<std::shared_ptr<ExprNode>>(); auto rhs_expr =
rhs.cast<std::shared_ptr<ExprNode>>(); BinaryOpNode::BinaryOpToken op =
BinaryOpASTHelper(name); return std::make_shared<BinaryOpNode>(op, lhs_expr,
rhs_expr);
}
static std::shared_ptr<UnaryOpNode> convertUnaryExpr(py::object opname, const
py::object &operand) { std::string name = opname.cast<std::string>(); auto
operand_expr = operand.cast<std::shared_ptr<ExprNode>>();
  UnaryOpNode::UnaryOpToken op = UnaryOpASTHelper(name);
  return std::make_shared<UnaryOpNode>(op, operand_expr);
}
static std::shared_ptr<LValNode> convertNameLVal(py::object name) {
  return std::make_shared<LValNode>(name.attr("id").cast<std::string>());
}
static std::shared_ptr<LValInitNode> convertNameLValInit(py::object name) {
  return std::make_shared<LValInitNode>(name.attr("id").cast<std::string>());
}
static std::shared_ptr<VariableNode> convertNameExp(py::object name) {
  return std::make_shared<VariableNode>(name.attr("id").cast<std::string>());
}

static std::shared_ptr<ConstantNode> convertConstant(py::object constant, const
py::object& type) { BasicType *constType =
getBasicTyFromString(type.cast<std::string>()); if (constType ==
BasicType::I32_TYPE) { int constValue = constant.cast<int>(); return
std::make_shared<ConstantNode>(constValue); } else if (constType ==
BasicType::BOOL_TYPE) { bool constValue = constant.cast<bool>(); return
std::make_shared<ConstantNode>(constValue); } else { float constValue =
constant.cast<float>(); return std::make_shared<ConstantNode>(constValue);
  }
}

static std::shared_ptr<FunctionDefNode>
convertFunctionDef(py::object name, const py::list &arguments,
                   const py::list &stmt_list) {
  auto funcName = name.cast<std::string>();
  std::vector<std::string> args;
  for (const auto &arg : arguments) {
    auto arg_name = arg.attr("id").cast<std::string>();
    args.push_back(arg_name);
  }
  auto proto = std::make_shared<PrototypeNode>(funcName, args);
  std::vector<std::shared_ptr<StmtNode>> body;
  for (const auto &stmt : stmt_list) {
    auto stmt_node = stmt.cast<std::shared_ptr<StmtNode>>();
    body.push_back(stmt_node);
  }
  return std::make_shared<FunctionDefNode>(proto, body);
}

static std::shared_ptr<ReturnNode> convertReturn(py::object value) {
  auto return_value = value.cast<std::shared_ptr<ExprNode>>();
  return std::make_shared<ReturnNode>(return_value);
}
};

std::shared_ptr<Module> exportModule(py::object ast, py::list &typed_args) {
DEBUG("Export module...")
auto _ast = ast.cast<std::shared_ptr<AINLAst>>();
std::map<std::string, Type *> typedInput;
std::vector<BasicType *> inputDtype;
std::vector<std::vector<int>> inputShape;
std::vector<std::vector<int>> inputStride;
for (const auto &arg : typed_args) {
  auto tupleArg = arg.cast<py::tuple>();
  auto argName = py::str(tupleArg[0]).cast<std::string>();
  auto typeTuple = tupleArg[1].cast<py::tuple>();
  auto dtypeName = py::str(typeTuple[0]).cast<std::string>();
  auto shapeTuple = typeTuple[1].cast<py::tuple>();

  std::vector<int> shapes;
  std::vector<int> strides;
  for (const auto &shape : shapeTuple) {
    shapes.push_back(shape.cast<int>());
    strides.push_back(1);
  }

  auto basicType = getBasicTyFromString(dtypeName);
  inputDtype.push_back(basicType);

  Type *tensorType = basicType;

  for (int i = shapes.size() - 1; i >= 0; i--) {
    int len = shapes[i];
    tensorType = new TensorType(len, tensorType, strides[i]);
  }

  typedInput[argName] = tensorType;
  DEBUG(argName)
}
DEBUG("Begin type inference...")
_ast->typeInfer(typedInput);
DEBUG("Finish type inference...")
DEBUG("Begin IR lowering...")
auto module = _ast->irLowering();
DEBUG("Finish IR lowering...")

//
好像是pybind11的链接时linkage的问题：如果对transform的调用写在另一个.cpp文件里(或者打包到另一个.so里)，
// 对static成员变量的访问会出问题(会使用另一个.cpp文件里的static成员变量实例)...
// 暂时先放在这里吧
module->transform("Autograd");
return module;
}

AINLAst wrapAst(std::shared_ptr<ModuleNode> module) { return AINLAst(module); }

PYBIND11_MODULE(AINL_FRONTEND, m) {
m.attr("__version__") = "0.1";

py::class_<AstNode, std::shared_ptr<AstNode>>(m, "AstNode");

py::class_<ModuleNode, AstNode, std::shared_ptr<ModuleNode>>(
    m, "ModuleNode", py::dynamic_attr())
    .def(py::init<>());

py::class_<StmtNode, AstNode, std::shared_ptr<StmtNode>>(m, "StmtNode")
    .def(py::init<>());

py::class_<AssignNode, StmtNode, std::shared_ptr<AssignNode>>(
    m, "AssignNode", py::dynamic_attr())
    .def(py::init<>());

py::class_<ExprNode, AstNode, std::shared_ptr<ExprNode>>(m, "ExprNode")
    .def(py::init<>());

py::class_<BinaryOpNode, ExprNode, std::shared_ptr<BinaryOpNode>>(
    m, "BinaryOpNode", py::dynamic_attr())
    .def(py::init<>());

py::class_<UnaryOpNode, ExprNode, std::shared_ptr<UnaryOpNode>>(
    m, "UnaryOpNode", py::dynamic_attr())
    .def(py::init<>());

py::class_<VariableNode, ExprNode, std::shared_ptr<VariableNode>>(
    m, "VariableNode")
    .def(py::init<>());
py::class_<LValNode, ExprNode, std::shared_ptr<LValNode>>(m, "LValNode")
    .def(py::init<>());
py::class_<LValInitNode, ExprNode, std::shared_ptr<LValInitNode>>(
    m, "LValInitNode")
    .def(py::init<>());
py::class_<ConstantNode, ExprNode, std::shared_ptr<ConstantNode>>(
    m, "ConstantNode")
    .def(py::init<>());
py::class_<PrototypeNode, AstNode, std::shared_ptr<PrototypeNode>>(
    m, "PrototypeNode")
    .def(py::init<>());

py::class_<FunctionDefNode, StmtNode, std::shared_ptr<FunctionDefNode>>(
    m, "FunctionDefNode", py::dynamic_attr())
    .def(py::init<>());

py::class_<ReturnNode, StmtNode, std::shared_ptr<ReturnNode>>(
    m, "ReturnNode", py::dynamic_attr())
    .def(py::init<>());

py::class_<AstTransformer>(m, "AstTransformer")
    .def(py::init<>())
    .def_static("convertModule", &AstTransformer::convertModule)
    .def_static("convertAssign", &AstTransformer::convertAssign)
    .def_static("convertExpr", &AstTransformer::convertExpr)
    .def_static("convertBinaryExpr", &AstTransformer::convertBinaryExpr)
    .def_static("convertUnaryExpr", &AstTransformer::convertUnaryExpr)
    .def_static("convertNameExp", &AstTransformer::convertNameExp)
    .def_static("convertNameLVal", &AstTransformer::convertNameLVal)
    .def_static("convertNameLValInit", &AstTransformer::convertNameLValInit)
    .def_static("convertConstant", &AstTransformer::convertConstant)
    .def_static("convertFunctionDef", &AstTransformer::convertFunctionDef)
    .def_static("convertReturn", &AstTransformer::convertReturn);

py::class_<AINLAst, std::shared_ptr<AINLAst>>(m, "AINLAst").def(py::init<>());

m.def("wrapAst", [](const py::object &module) {
  auto mod = module.cast<std::shared_ptr<ModuleNode>>();
  return wrapAst(mod);
});

m.def("exportModule", [](py::object ast, py::list &typed_args) {
  return exportModule(ast, typed_args);
});

py::class_<Value, std::shared_ptr<Value>>(m, "Value");
py::class_<Module, Value, std::shared_ptr<Module>>(m, "Module",
                                                   py::dynamic_attr())
    .def(py::init<>());
    // .def("transform", [](Module& self, std::string transform_name) {
    //  return self.transform(transform_name);
    // });
}
*/

void initAst(py::module_ &m);

#endif // AINL_SRC_INCLUDE_AST_BINDING_H