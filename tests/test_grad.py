import numpy as np
import ailang as al

@al.grad
def g(x):
    return x

a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = al.from_numpy(a)
iree_result = g(b)
print("Result: ", iree_result)
