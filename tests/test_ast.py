import ailang
from ailang import parse_pycallable
from ailang import ModuleNode, VarDefNode, VarNode, ConstantNode, TupleNode

class TestVarDef:
    def test_variable(self):
        code_str = "x = y"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode([
            VarDefNode(
                [VarNode("x")],
                VarNode("y")
            )
        ])
        assert ast.match(ref_ast)

    def test_constant(self):
        code_str = "x = 1"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode([
            VarDefNode(
                [VarNode("x")],
                ConstantNode("1")
            )
        ])
        assert ast.match(ref_ast)

    def test_tuple(self):
        code_str = "x, y = z"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode([
            VarDefNode(
                [TupleNode([VarNode("x"), VarNode("y")])],
                VarNode("z")
            )
        ])
        assert ast.match(ref_ast)

    def test_sequential_vardef(self):
        code_str = "x = y = z"
        ast = parse_pycallable(code_str)
        ref_ast = ModuleNode([
            VarDefNode(
                [VarNode("x"), VarNode("y")],
                VarNode("z")
            )
        ])
        assert ast.match(ref_ast)








