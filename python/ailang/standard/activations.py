import ailang as al

from typing import List, Optional
from .common import _tensor_member_fn


@_tensor_member_fn
@al.to_static
def relu(x: al.array) -> al.array:
    """Computes the relu of a tensor."""
    return al.prim.relu(x)


@_tensor_member_fn
@al.to_static
def sigmoid(x: al.array) -> al.array:
    """Computes the relu of a tensor."""
    exp = al.prim.exp(al.prim.neg(x))
    import numpy as np
    one = al.prim.broadcast_to(al.from_numpy(np.array(1)), tuple(x.shape))
    return al.div(one, al.add(one, exp))


@_tensor_member_fn
@al.to_static
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
