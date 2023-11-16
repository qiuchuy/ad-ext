import ailang as al

from typing import Union, Tuple, Callable
from ailang import compile_ast, Tensor, TensorType
from ailang import (
    ModuleNode,
    BindNode,
    VarNode,
    ConstantNode,
    TupleNode,
    BinaryOpNode,
    UnaryOpNode,
    CallNode,
    CompareNode,
    WhileNode,
    FunctionDefNode,
    ReturnNode,
)


class TestBind:
    def test_return_once(self):
        def f(x):
            return x

        a = al.tensor((1, 2, 3), "Int")
        typed_ast = compile_ast(f, a)
        ref_ast = ModuleNode(
            [
                FunctionDefNode(
                    "f", ["x"], [ReturnNode(VarNode("x", TensorType((1, 2, 3), "Int")))]
                )
            ]
        )
        assert typed_ast.match(ref_ast)

    def test_return_later(self):
        def f(x):
            y = x
            return y

        a = al.tensor((1, 2, 3), "Float")
        typed_ast = compile_ast(f, a)
        ref_ast = ModuleNode(
            [
                FunctionDefNode(
                    "f",
                    ["x"],
                    [
                        BindNode(
                            [VarNode("y", TensorType((1, 2, 3), "Float"))],
                            VarNode("x", TensorType((1, 2, 3), "Float")),
                        ),
                        ReturnNode(VarNode("y", TensorType((1, 2, 3), "Float"))),
                    ],
                )
            ]
        )
        assert typed_ast.match(ref_ast)

    def test_type_comparison(self):
        def f(x, y):
            return x + y

        a = al.tensor((1, 2, 3), "Float")
        b = al.tensor((1, 2, 3), "Int")
        typed_ast = compile_ast(f, a, b)
        ref_ast = ModuleNode(
            [
                FunctionDefNode(
                    "f",
                    ["x", "y"],
                    [
                        ReturnNode(
                            BinaryOpNode(
                                "Add",
                                VarNode("x", TensorType((1, 2, 3), "Float")),
                                VarNode("y", TensorType((1, 2, 3), "Int")),
                                TensorType((1, 2, 3), "Float"),
                            )
                        )
                    ],
                )
            ]
        )
        assert typed_ast.match(ref_ast)

    def test_librarycall(self):
        def f(x, y):
            return al.matmul(x, y)

        a = al.tensor((1, 2), "Float")
        b = al.tensor((2, 3), "Float")
        typed_ast = compile_ast(f, a, b)
        ref_ast = ModuleNode(
            [
                FunctionDefNode(
                    "f",
                    ["x", "y"],
                    [
                        ReturnNode(
                            CallNode(
                                VarNode("al::matmul"),
                                [
                                    VarNode("x", TensorType((1, 2), "Float")),
                                    VarNode("y", TensorType((2, 3), "Float")),
                                ],
                                TensorType((1, 3), "Float"),
                            )
                        )
                    ],
                )
            ]
        )
        assert typed_ast.match(ref_ast)
