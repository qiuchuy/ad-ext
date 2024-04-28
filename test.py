import numpy as np
import ailang as al
print("============")
a = np.random.randn(2,2)
b = al.from_numpy(a)
c = al.cos(b)
print(c)
