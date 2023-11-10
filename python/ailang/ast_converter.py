import gast
import inspect
import textwrap
from typing import Callable, Union

from ailang import AstTransformer, ModuleNode


class SymbolTable:
    def __init__(self, parent=None):
        self.parent = parent
        self.child = []
        self.env = []

    def resolve(self):
        if self.parent is not None:
            idx = self.parent.child.index(self)
            if idx != (len(self.parent.child) - 1):
                return self.parent.child[idx + 1]
            else:
                return self.parent.resolve()
        else:
            return self

    def insert(self, variable):
        self.env.append(variable)

    def query(self, name):
        if name in self.env:
            return True
        else:
            while self.parent is not None:
                return self.parent.query(name)
            return False

    @classmethod
    def add_scope(cls, self):
        new_scope = cls(self)
        self.child.append(new_scope)
        return new_scope


class TransformerVisitor(gast.NodeVisitor):
    """
    Lowering Python AST into AINL Ast
    """

    def __init__(self):
        super().__init__()
        self.root = None
        self.transformer = AstTransformer()
        self.env = SymbolTable()

    def visit_Module(self, node):
        stmt_list = []
        for stmt in node.body:
            method_name = "visit_" + stmt.__class__.__name__
            stmt_node = getattr(self, method_name)(stmt)
            stmt_list.append(stmt_node)
        module = self.transformer.convert_Module(stmt_list)
        module.stmts = stmt_list
        self.root = module
        return module

    def visit_Assign(self, node):
        target_list = []
        for target in node.targets:
            method_name = "visit_" + target.__class__.__name__
            target_list.append(getattr(self, method_name)(target))
        source = node.value
        method_name = "visit_" + source.__class__.__name__
        src = getattr(self, method_name)(source)

        assign = self.transformer.convert_Assign(target_list, src)
        assign.targets = target_list
        assign.src = src
        return assign

    def visit_Tuple(self, node):
        elts = []
        for elt in node.elts:
            method_name = "visit_" + elt.__class__.__name__
            elts.append(getattr(self, method_name)(elt))
        tuple_node = self.transformer.convert_Tuple(elts)
        tuple_node.elts = elts
        return tuple_node

    def visit_FunctionDef(self, node):
        # [TODO] handle class method
        arg_list = []
        for argument in node.args.args:
            arg_list.append(argument.id)
        body_list = []
        for body in node.body:
            method_name = "visit_" + body.__class__.__name__
            body_list.append(getattr(self, method_name)(body))
        name = node.name
        func_def = self.transformer.convert_FunctionDef(name, arg_list, body_list)
        func_def.name = name
        func_def.args = arg_list
        func_def.body = body_list
        return func_def

    def visit_Return(self, node):
        method_name = "visit_" + node.value.__class__.__name__
        value = getattr(self, method_name)(node.value)
        return_node = self.transformer.convert_Return(value)
        return_node.value = value
        return return_node

    def visit_Name(self, node):
        return self.transformer.convert_Name(node.id)

    def visit_Constant(self, node):
        return self.transformer.convert_Constant(str(node.value))

    def visit_BinOp(self, node):
        lhs_method = "visit_" + node.left.__class__.__name__
        lhs = getattr(self, lhs_method)(node.left)
        rhs_method = "visit_" + node.right.__class__.__name__
        rhs = getattr(self, rhs_method)(node.right)
        opname = node.op.__class__.__name__
        binop = self.transformer.convert_BinOp(opname, lhs, rhs)
        binop.lhs = lhs
        binop.rhs = rhs
        return binop

    def visit_UnaryOp(self, node):
        method = "visit_" + node.operand.__class__.__name__
        operand = getattr(self, method)(node.operand)
        opname = node.op.__class__.__name__
        unaryop = self.transformer.convert_UnaryOp(opname, operand)
        unaryop.operand = operand
        return unaryop

    def visit_Call(self, node):
        method = "visit_" + node.func.__class__.__name__
        func = getattr(self, method)(node.func)
        args = []
        for arg in node.args:
            method_name = "visit_" + arg.__class__.__name__
            args.append(getattr(self, method_name)(arg))
        # [TODO] Support Keyword Args
        call = self.transformer.convert_Call(func, args)
        call.func = func
        call.args = args
        return call

    def visit_Attribute(self, node):
        namespace = node.value.id
        attr = node.attr
        return self.transformer.convert_Attribute(namespace + "::" + attr)

    def visit_Expr(self, node):
        value_method = "visit_" + node.value.__class__.__name__
        return getattr(self, value_method)(node.value)

    def visit_Compare(self, node):
        left = node.left
        left_method = "visit_" + left.__class__.__name__
        left_node = getattr(self, left_method)(left)
        ops = []
        for op in node.ops:
            ops.append(op.__class__.__name__)
        comparators = []
        for comparator in node.comparators:
            comparator_method = "visit_" + comparator.__class__.__name__
            comparator_node = getattr(self, comparator_method)(comparator)
            comparators.append(comparator_node)
        compare = self.transformer.convert_Compare(left_node, ops, comparators)
        compare.left = left_node
        compare.ops = ops
        compare.comparators = comparators
        return compare

    def visit_While(self, node):
        test_method = "visit_" + node.test.__class__.__name__
        test = getattr(self, test_method)(node.test)
        body = []
        for stmt in node.body:
            method_name = "visit_" + stmt.__class__.__name__
            body.append(getattr(self, method_name)(stmt))
        while_node = self.transformer.convert_While(test, body)
        while_node.test = test
        while_node.body = body
        return while_node

    def transform(self, tree) -> ModuleNode:
        self.visit(tree)
        return self.root


def parse_pycallable(source: Union[str, Callable], verbose: bool = False):
    """
    Parse a python callable into ModuleNode in AILang Ast
    :param source: a python callable or its string form
    :param verbose: print debug information
    :return: a ModuleNode in AILang Ast
    """
    tree = None
    if isinstance(source, str):
        tree = gast.parse(source)
    else:
        tree = gast.parse(textwrap.dedent(inspect.getsource(source)))
    if verbose:
        print(gast.dump(tree, indent=4))
    transformer = TransformerVisitor()
    ast = transformer.transform(tree)
    return ast
