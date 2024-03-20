import ailang as al
import numpy as np

a = np.array([1, 2])
b = al.from_numpy(a)
print(b)
print(b.ndim)
print(b.shape)
