#include "ast_binding.h"

void initAst(py::module_ &m) {
    py::class_<ASTNode, std::shared_ptr<ASTNode>>(m, "ASTNode", py::dynamic_attr())
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
        .def(py::init<std::vector<Stmt>>());


    py::class_<VarDefNode, StmtNode, std::shared_ptr<VarDefNode>>(
        m, "VarDefNode", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<std::vector<Expr>, Expr>());


    py::class_<TupleNode, ExprNode, std::shared_ptr<TupleNode>>(
            m, "TupleNode", py::dynamic_attr())
            .def(py::init<>())
            .def(py::init<std::vector<Expr>>());

    py::class_<BinaryOpNode, ExprNode, std::shared_ptr<BinaryOpNode>>(
        m, "BinaryOpNode", py::dynamic_attr())
        .def(py::init<>());

    py::class_<UnaryOpNode, ExprNode, std::shared_ptr<UnaryOpNode>>(
        m, "UnaryOpNode", py::dynamic_attr())
        .def(py::init<>());

    py::class_<VarNode, ExprNode, std::shared_ptr<VarNode>>(m, "VarNode")
        .def(py::init<>())
        .def(py::init<std::string>());

    py::class_<ConstantNode, ExprNode, std::shared_ptr<ConstantNode>>(
        m, "ConstantNode")
        .def(py::init<>())
        .def(py::init<std::string>());


    py::class_<FunctionDefNode, StmtNode, std::shared_ptr<FunctionDefNode>>(
        m, "FunctionDefNode", py::dynamic_attr())
        .def(py::init<>());

    py::class_<ReturnNode, StmtNode, std::shared_ptr<ReturnNode>>(
        m, "ReturnNode", py::dynamic_attr())
        .def(py::init<>());

    py::class_<WhileLoopNode, StmtNode, std::shared_ptr<WhileLoopNode>>(
        m, "WhileLoopNode", py::dynamic_attr())
        .def(py::init<>());

    py::class_<IfNode, StmtNode, std::shared_ptr<IfNode>>(m, "IfNode",
                                                          py::dynamic_attr())
        .def(py::init<>());

    py::class_<CallNode, ExprNode, std::shared_ptr<CallNode>>(
        m, "CallNode", py::dynamic_attr())
        .def(py::init<>());

    py::class_<AstTransformer>(m, "AstTransformer")
        .def(py::init<>())
        .def_static("convert_Module", &AstTransformer::convertModule,
                    py::return_value_policy::reference)
        .def_static("convert_FunctionDef", &AstTransformer::convertFunctionDef,
                    py::return_value_policy::reference)
        .def_static("convert_Assign", &AstTransformer::convertVarDef,
                    py::return_value_policy::reference)
        .def_static("convert_Name", &AstTransformer::convertVar,
                    py::return_value_policy::reference)
        .def_static("convert_Constant", &AstTransformer::convertConstant,
                    py::return_value_policy::reference)
        .def_static("convert_Tuple", &AstTransformer::convertTuple,
                    py::return_value_policy::reference);
}