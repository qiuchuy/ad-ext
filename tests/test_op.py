import ailang as al
import numpy as np

a = np.random.randn(2, 2)
b = al.from_numpy(a)
c = al.flatten(b)
print(b)
print(c)