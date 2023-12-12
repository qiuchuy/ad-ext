import ailang as al

a = al.tensor((1, 2), "Float")
b = al.tensor((2, 3), "Float")
c = al.tensor((1,3,224,224), "Float") 

def f_matmul(x, y):
    return al.matmul(x, y)

def f_relu(x):
    return al.relu(x)


def f_maxcpool2d(x):
    return al.maxpool2d(x)

def f_convolution(x):
    return al.convolution(x)

def f_transpose(x):
    return al.transpose(x)


print(al.compile_ir(f_matmul, a, b))
# ir = al.compile_ast(f,a,b)
print(al.compile_ir(f_relu,a))

print(al.compile_ir(f_maxcpool2d,c))

print(al.compile_ir(f_convolution,c))

print(al.compile_ir(f_transpose,a))