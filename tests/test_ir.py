import ailang as al

a = al.tensor((1, 2), "Float")
b = al.tensor((2, 3), "Float")


def f(x, y):
    return al.matmul(x, y)


ir = al.compile_ir(f, a, b)
print(ir)
