import ailang as al

from typing import List
from .common import _tensor_member_fn


@_tensor_member_fn
@al.jit
def relu(x: al.array) -> al.array:
    """Computes the relu of a tensor."""
    return al.prim.relu(x)
