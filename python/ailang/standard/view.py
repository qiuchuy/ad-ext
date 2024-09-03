import ailang as al

from typing import List
from .common import _tensor_member_fn


@_tensor_member_fn
@al.jit
def transpose(x: al.array) -> al.array:
    """Transposes a tensor."""
    return al.prim.transpose(x)
