import ailang as al
from typing import List


def softmax(x: al.array, dim: List[int] = None) -> al.array:
    if dim is None:
        shape = x.shape
        dim = [len(shape) - 1]
    x_max = al.prim.max(x, dim, True)
    x_max_broad = al.prim.broadcast_to(x_max, tuple(x.shape))
    sub = al.prim.add(x, al.prim.neg(x_max_broad))
    exp_z = al.prim.exp(sub)
    sum_exp_z = al.prim.sum(exp_z, dim, True)
    sum_exp_z_broad = al.prim.broadcast_to(sum_exp_z, tuple(x.shape))
    return al.prim.div(exp_z, sum_exp_z_broad)
