import ailang as al

from typing import List, Tuple,Union
from .common import _tensor_member_fn, flatten_pytree


@_tensor_member_fn
@al.jit
def transpose(x: al.array) -> al.array:
    """Transposes a tensor."""
    return al.prim.transpose(x)

@al.jit
def cat(arrays: Union[Tuple[al.array], List[al.array]], dim: int) -> al.array:
    """Concatenates the given sequence of seq tensors in the given dimension."""
    return al.prim.cat(arrays, dim)
