#include "ast_binding.h"
#include "tensor.h"
#include "type_infer.h"
#include "utils.h"
#include "visitor.h"

void initAST(py::module_ &m) {
    py::class_<ASTNode, std::shared_ptr<ASTNode>>(m, "ASTNode",
                                                  py::dynamic_attr())
        .def(py::init<>())
        .def("__str__", &ASTNode::getName)
        .def("match", &ASTNode::match);

    py::class_<StmtNode, ASTNode, std::shared_ptr<StmtNode>>(m, "StmtNode")
        .def(py::init<>());

    py::class_<ExprNode, ASTNode, std::shared_ptr<ExprNode>>(m, "ExprNode")
        .def(py::init<>());

    py::class_<ModuleNode, StmtNode, std::shared_ptr<ModuleNode>>(
        m, "ModuleNode", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<std::vector<Stmt>>())
        .def(
            "type_infer",
            [](const Module &self, const std::vector<std::string> &argNames,
               const py::args &args) {
                assert(args.size() == argNames.size());
                std::vector<TypePtr> argTypes;
                for (const auto &arg : args) {
                    if (arg.cast<TensorPtr>()) {
                        auto tensor = arg.cast<TensorPtr>();
                        argTypes.push_back(tensor->getType());
                    }
                    if (py::isinstance<py::int_>(arg)) {
                        argTypes.push_back(IntTypePtr::get());
                    }
                    if (py::isinstance<py::float_>(arg)) {
                        argTypes.push_back(FloatTypePtr ::get());
                    }
                    if (py::isinstance<py::bool_>(arg)) {
                        argTypes.push_back(BoolTypePtr::get());
                    }
                }
                auto visitor = std::make_unique<TypeInfer>(argNames, argTypes);
                auto clonedModule = std::make_shared<ModuleNode>(*self);
                clonedModule->accept(visitor.get());
                return clonedModule;
            },
            py::return_value_policy::reference);

    py::class_<BindNode, StmtNode, std::shared_ptr<BindNode>>(
        m, "BindNode", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<std::vector<Expr>, Expr>());

    py::class_<TupleNode, ExprNode, std::shared_ptr<TupleNode>>(
        m, "TupleNode", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<std::vector<Expr>>());

    py::class_<BinaryOpNode, ExprNode, std::shared_ptr<BinaryOpNode>>(
        m, "BinaryOpNode", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init(
            [](const std::string &op, const Expr &op1, const Expr &op2) {
                return std::make_shared<BinaryOpNode>(BinaryOpASTHelper(op),
                                                      op1, op2);
            }))
        .def(py::init([](const std::string &op, const Expr &op1,
                         const Expr &op2, const TypePtr &type) {
            return std::make_shared<BinaryOpNode>(BinaryOpASTHelper(op), op1,
                                                  op2, type);
        }));

    py::class_<UnaryOpNode, ExprNode, std::shared_ptr<UnaryOpNode>>(
        m, "UnaryOpNode", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init([](const std::string &op, const Expr &value) {
            return std::make_shared<UnaryOpNode>(UnaryOpASTHelper(op), value);
        }));

    py::class_<VarNode, ExprNode, std::shared_ptr<VarNode>>(m, "VarNode")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def(py::init<std::string, TypePtr>());
    py::class_<VarDefNode, ExprNode, std::shared_ptr<VarDefNode>>(m,
                                                                  "VarDefNode")
        .def(py::init<>())
        .def(py::init<std::string>());

    py::class_<ConstantNode, ExprNode, std::shared_ptr<ConstantNode>>(
        m, "ConstantNode")
        .def(py::init<>())
        .def(py::init<std::string>());

    py::class_<FunctionDefNode, StmtNode, std::shared_ptr<FunctionDefNode>>(
        m, "FunctionDefNode", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<std::string, std::vector<std::string>,
                      std::vector<Stmt>>());

    py::class_<ReturnNode, StmtNode, std::shared_ptr<ReturnNode>>(
        m, "ReturnNode", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<Expr>());

    py::class_<CompareNode, ExprNode, std::shared_ptr<CompareNode>>(
        m, "CompareNode")
        .def(py::init<>())
        .def(py::init([](const Expr &left, const std::vector<std::string> &ops,
                         const std::vector<Expr> &comparators) {
            std::vector<CompareNode::CompareOpKind> iops;
            for (const auto &op : ops)
                iops.push_back(CompareOpASTHelper(op));
            return std::make_shared<CompareNode>(left, iops, comparators);
        }));

    py::class_<WhileNode, StmtNode, std::shared_ptr<WhileNode>>(
        m, "WhileNode", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<Expr, std::vector<Stmt>>());

    py::class_<IfNode, StmtNode, std::shared_ptr<IfNode>>(m, "IfNode",
                                                          py::dynamic_attr())
        .def(py::init<>());

    py::class_<CallNode, ExprNode, std::shared_ptr<CallNode>>(
        m, "CallNode", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<Expr, std::vector<Expr>>())
        .def(py::init<Expr, std::vector<Expr>, TypePtr>());

    py::class_<AstTransformer>(m, "AstTransformer")
        .def(py::init<>())
        .def_static("convert_Module", &AstTransformer::convertModule,
                    py::return_value_policy::reference)
        .def_static("convert_FunctionDef", &AstTransformer::convertFunctionDef,
                    py::return_value_policy::reference)
        .def_static("convert_Assign", &AstTransformer::convertBind,
                    py::return_value_policy::reference)
        .def_static("convert_Name", &AstTransformer::convertVar,
                    py::return_value_policy::reference)
        .def_static("convert_NameDef", &AstTransformer::convertVarDef,
                    py::return_value_policy::reference)
        .def_static("convert_Constant", &AstTransformer::convertConstant,
                    py::return_value_policy::reference)
        .def_static("convert_Tuple", &AstTransformer::convertTuple,
                    py::return_value_policy::reference)
        .def_static("convert_BinOp", &AstTransformer::convertBinaryOp,
                    py::return_value_policy::reference)
        .def_static("convert_UnaryOp", &AstTransformer::convertUnaryOp,
                    py::return_value_policy::reference)
        .def_static("convert_Call", &AstTransformer::convertCall,
                    py::return_value_policy::reference)
        .def_static("convert_Attribute", &AstTransformer::convertVar,
                    py::return_value_policy::reference)
        .def_static("convert_Compare", &AstTransformer::convertCompare,
                    py::return_value_policy::reference)
        .def_static("convert_While", &AstTransformer::convertWhile,
                    py::return_value_policy::reference)
        .def_static("convert_Return", &AstTransformer::convertReturn,
                    py::return_value_policy::reference);
}