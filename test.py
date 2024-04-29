import numpy as np
import ailang as al
print("============")
a = np.random.randn(2,2)
b = al.from_numpy(a)
c = al.cos(b)

a1 = np.random.randn(2,2)
b1 = al.from_numpy(a1)
c1 = al.cos(b1)

print(c)
print(c1)

d = al.Add(c,c1)
print(d)

