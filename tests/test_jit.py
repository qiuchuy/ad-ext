import numpy as np
import ailang as al

def f(x, y):
    z = al.transpose(y)
    return al.matmul(x, z)

a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 2], [3, 4]])
c = al.from_numpy(a)
d = al.from_numpy(b)
module = al.jit(f, (c, d, ))
print(module)