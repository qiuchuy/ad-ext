import numpy as np
import ailang as al


@al.grad
def g(x, y):
    return al.matmul(x, y)

a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = np.array([[5, 6], [7, 8]], dtype=np.float32)
c = al.from_numpy(a)
d = al.from_numpy(b)
iree_result = g(c, d)
print("Result: ", iree_result)
