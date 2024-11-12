import ailang as al
from typing import List


def softmax(x: al.array, dim: List[int] = None) -> al.array:
    if dim is None:
        shape = x.shape
        dim = [len(shape) - 1]
    exp_z = al.exp(x)
    sum_exp_z = al.sum(exp_z, dim, True)
    return al.div(exp_z, sum_exp_z)
