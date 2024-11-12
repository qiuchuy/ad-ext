import ailang as al

from typing import List, Tuple, Union, Optional
from .common import _tensor_member_fn, flatten_pytree


@_tensor_member_fn
@al.jit
def transpose(x: al.array, perm: Optional[List[int]] = None) -> al.array:
    """Transposes a tensor."""
    if perm is None:
        perm = []
        x_len = len(x.shape)
        for i in range(x_len):
            perm.append(x_len - i - 1)
    return al.prim.transpose(x, perm)



@al.jit
def cat(arrays: Union[Tuple[al.array], List[al.array]], dim: int) -> al.array:
    """Concatenates the given sequence of seq tensors in the given dimension."""
    return al.prim.cat(arrays, dim)


@_tensor_member_fn
@al.jit
def broadcast_to(x: al.array, shape) -> al.array:
    """Broadcasts a tensor to a new shape."""
    return al.prim.broadcast_to(x, shape)
