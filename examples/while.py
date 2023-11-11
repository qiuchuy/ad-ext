import ailang as al
from ailang import compile_ast

a = al.tensor((1, 2, 3), "Float")


def f(x):
    i = 0
    while i < 10:
        y = x + 1
    return y


ast = compile_ast(f, a)
print(ast)
