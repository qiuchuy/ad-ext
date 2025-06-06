import ailang as al

from typing import List
from .common import _tensor_member_fn


@_tensor_member_fn
@al.to_static
def mean(x: al.array, dim: List[int] = None) -> al.array:
    """Computes the mean of a tensor."""
    if dim is None:
        shape = x.shape
        dim = list(range(len(shape)))
    return al.prim.mean(x, dim)


@_tensor_member_fn
@al.to_static
def sum(x: al.array, dim: List[int] = None, keepdims=False) -> al.array:
    """Computes the sum of a tensor."""
    if dim is None:
        shape = x.shape
        dim = list(range(len(shape)))
    return al.prim.sum(x, dim, keepdims)


@_tensor_member_fn
@al.to_static
def max(x: al.array, dim: List[int] = None, keepdims=False) -> al.array:
    """Computes the sum of a tensor."""
    if dim is None:
        shape = x.shape
        dim = list(range(len(shape)))
    return al.prim.max(x, dim, keepdims)


@_tensor_member_fn
@al.to_static
def var(x: al.array, dim: List[int], ddof) -> al.array:
    """Computes the var of a tensor."""
    if dim is None:
        shape = x.shape
        dim = list(range(len(shape)))
    return al.prim.var(x, dim, ddof)
