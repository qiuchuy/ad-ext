import ailang as al
from ailang import grad
a = al.randn((2, 2))
b = al.randn((2, 2))
@grad
def forward(a, b): return al.sum(al.matmul(a, b))
c, da, db = forward(a, b)
print(c)
print(da)
print(db)