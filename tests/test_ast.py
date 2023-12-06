import ailang
from ailang import parse_pycallable
from ailang import (
    ModuleNode,
    BindNode,
    VarNode,
    VarDefNode,
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
    def test_variable(self):
        code_str = "x = y"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode([BindNode([VarDefNode("x")], VarNode("y"))])
        assert ast.match(ref_ast)

    def test_constant(self):
        code_str = "x = 1"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode([BindNode([VarDefNode("x")], ConstantNode("1"))])
        assert ast.match(ref_ast)

    def test_tuple(self):
        code_str = "x, y = z"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [BindNode([TupleNode([VarDefNode("x"), VarDefNode("y")])], VarNode("z"))]
        )
        assert ast.match(ref_ast)

    def test_sequential(self):
        code_str = "x = y = z"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [BindNode([VarDefNode("x"), VarDefNode("y")], VarNode("z"))]
        )
        assert ast.match(ref_ast)

    def test_binop_variable(self):
        code_str = "x = y + z"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                BindNode(
                    [VarDefNode("x")], BinaryOpNode("Add", VarNode("y"), VarNode("z"))
                )
            ]
        )
        assert ast.match(ref_ast)

    def test_binop_constant(self):
        code_str = "x = y + 1"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                BindNode(
                    [VarDefNode("x")],
                    BinaryOpNode("Add", VarNode("y"), ConstantNode("1")),
                )
            ]
        )
        assert ast.match(ref_ast)

    def test_uop_variable(self):
        code_str = "x = -y"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [BindNode([VarDefNode("x")], UnaryOpNode("USub", VarNode("y")))]
        )
        assert ast.match(ref_ast)

    def test_uop_constant(self):
        code_str = "x = -1"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [BindNode([VarDefNode("x")], UnaryOpNode("USub", ConstantNode("1")))]
        )
        assert ast.match(ref_ast)

    def test_attribute(self):
        code_str = "x = self.param"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode([BindNode([VarDefNode("x")], VarNode("self::param"))])
        assert ast.match(ref_ast)

    def test_call_variable(self):
        code_str = "x = f(a, b)"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                BindNode(
                    [VarDefNode("x")],
                    CallNode(VarNode("f"), [VarNode("a"), VarNode("b")]),
                )
            ]
        )
        assert ast.match(ref_ast)

    def test_call_constant(self):
        code_str = "x = f(a, 1)"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                BindNode(
                    [VarDefNode("x")],
                    CallNode(VarNode("f"), [VarNode("a"), ConstantNode("1")]),
                )
            ]
        )
        assert ast.match(ref_ast)

    def test_call_attribute(self):
        code_str = "x = self.f(a, 1)"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                BindNode(
                    [VarDefNode("x")],
                    CallNode(VarNode("self::f"), [VarNode("a"), ConstantNode("1")]),
                )
            ]
        )
        assert ast.match(ref_ast)

    def test_call_keywargs(self):
        code_str = "x = f(a, b=c)"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                BindNode(
                    [VarDefNode("x")],
                    CallNode(VarNode("f"), [VarNode("a")], {"b":VarNode("c")}),
                )
            ]
        )
        assert ast.match(ref_ast) 

    def test_call_keywargs_constant(self):
        code_str = "x = f(a, b=1)"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                BindNode(
                    [VarDefNode("x")],
                    CallNode(VarNode("f"), [VarNode("a")], {"b":ConstantNode("1")}),
                )
            ]
        )
        assert ast.match(ref_ast) 

    def test_call_keywargs_tuple(self):
        code_str = "x = f(a, b=(1, 2))"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                BindNode(
                    [VarDefNode("x")],
                    CallNode(VarNode("f"), [VarNode("a")], {"b":TupleNode([ConstantNode("1"), ConstantNode("2")])}),
                )
            ]
        )
        assert ast.match(ref_ast) 

    def test_compare_lt(self):
        code_str = "z = x < y"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                BindNode(
                    [VarDefNode("z")], CompareNode(VarNode("x"), ["Lt"], [VarNode("y")])
                )
            ]
        )
        assert ast.match(ref_ast)


class TestLoop:
    def test_while(self):
        code_str = "while i < j: \
                        y = x"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                WhileNode(
                    CompareNode(VarNode("i"), ["Lt"], [VarNode("j")]),
                    [BindNode([VarDefNode("y")], VarNode("x"))],
                )
            ]
        )
        assert ast.match(ref_ast)


class TestFunctionDef:
    def test_return(self):
        code_str = "def f(x):\
                        return x"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode([FunctionDefNode("f", ["x"], [ReturnNode(VarNode("x"))])])
        assert ast.match(ref_ast)

    def test_return_noreturn(self):
        code_str = "def f(x):\
                        y = x"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [FunctionDefNode("f", ["x"], [BindNode([VarNode("y")], VarNode("x"))])]
        )
        assert ast.match(ref_ast)

    def test_multiple_args(self):
        code_str = "def f(x, y):\n" "    z = x + y\n" "    return z"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                FunctionDefNode(
                    "f",
                    ["x", "y"],
                    [
                        BindNode(
                            [VarNode("z")],
                            BinaryOpNode("Add", VarNode("x"), VarNode("y")),
                        ),
                        ReturnNode(VarNode("z")),
                    ],
                )
            ]
        )
        assert ast.match(ref_ast)

        
