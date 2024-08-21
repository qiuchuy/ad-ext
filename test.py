import numpy as np
import ailang as al

a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = np.array([[1, 2], [3, 4]], dtype=np.float32)
print(b.strides)
c = al.from_numpy(a)
d = al.from_numpy(b)
print(d.strides)
e = al.transpose(d)
print("eval res", e)
