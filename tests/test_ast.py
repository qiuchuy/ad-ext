import ailang
from ailang import parse_pycallable
from ailang import (
    ModuleNode,
    VarDefNode,
    VarNode,
    ConstantNode,
    TupleNode,
    BinaryOpNode,
    UnaryOpNode,
    CallNode,
)


class TestVarDef:
    def test_variable(self):
        code_str = "x = y"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode([VarDefNode([VarNode("x")], VarNode("y"))])
        assert ast.match(ref_ast)

    def test_constant(self):
        code_str = "x = 1"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode([VarDefNode([VarNode("x")], ConstantNode("1"))])
        assert ast.match(ref_ast)

    def test_tuple(self):
        code_str = "x, y = z"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [VarDefNode([TupleNode([VarNode("x"), VarNode("y")])], VarNode("z"))]
        )
        assert ast.match(ref_ast)

    def test_sequential(self):
        code_str = "x = y = z"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode([VarDefNode([VarNode("x"), VarNode("y")], VarNode("z"))])
        assert ast.match(ref_ast)

    def test_binop_variable(self):
        code_str = "x = y + z"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                VarDefNode(
                    [VarNode("x")], BinaryOpNode("Add", VarNode("y"), VarNode("z"))
                )
            ]
        )
        assert ast.match(ref_ast)

    def test_binop_constant(self):
        code_str = "x = y + 1"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                VarDefNode(
                    [VarNode("x")], BinaryOpNode("Add", VarNode("y"), ConstantNode("1"))
                )
            ]
        )
        assert ast.match(ref_ast)

    def test_uop_variable(self):
        code_str = "x = -y"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [VarDefNode([VarNode("x")], UnaryOpNode("USub", VarNode("y")))]
        )
        assert ast.match(ref_ast)

    def test_uop_constant(self):
        code_str = "x = -1"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [VarDefNode([VarNode("x")], UnaryOpNode("USub", ConstantNode("1")))]
        )
        assert ast.match(ref_ast)

    def test_attribute(self):
        code_str = "x = self.param"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode([VarDefNode([VarNode("x")], VarNode("self::param"))])
        assert ast.match(ref_ast)

    def test_call_variable(self):
        code_str = "x = f(a, b)"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                VarDefNode(
                    [VarNode("x")], CallNode(VarNode("f"), [VarNode("a"), VarNode("b")])
                )
            ]
        )
        assert ast.match(ref_ast)

    def test_call_constant(self):
        code_str = "x = f(a, 1)"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode(
            [
                VarDefNode(
                    [VarNode("x")],
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
                VarDefNode(
                    [VarNode("x")],
                    CallNode(VarNode("self::f"), [VarNode("a"), ConstantNode("1")]),
                )
            ]
        )
        assert ast.match(ref_ast)


class TestWhile:
    pass
