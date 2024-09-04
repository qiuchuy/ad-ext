import ailang as al

from typing import List
from .common import _tensor_member_fn

@_tensor_member_fn
@al.jit
def add(x: al.array, y: al.array) -> al.array:
    """Add two tensors."""
    return al.prim.add(x, y)

@_tensor_member_fn
@al.jit
def conv2d(x: al.array, kernel: al.array, strides: List[int], padding: List[int], dilations: List[int]) -> al.array:
    """2D convolution."""
    return al.prim.conv2d(x, kernel, strides, padding, dilations)

@_tensor_member_fn
@al.jit
def relu(x: al.array) -> al.array:
    """ReLU."""
    return al.prim.relu(x)

@_tensor_member_fn
@al.jit
def batchnorm2d(x: al.array, scale: al.array, offset: al.array, mean: al.array, variance: al.array) -> al.array:
    """Batch normalization."""
    return al.prim.batchnorm2d(x, scale, offset, mean, variance)

@_tensor_member_fn
@al.jit
def maxpool2d(x: al.array) -> al.array:
    """Maxpool2d."""
    return al.prim.maxpool2d(x)

@_tensor_member_fn
@al.jit
def exp(x: al.array) -> al.array:
    """Exponential."""
    return al.prim.exp(x)

@_tensor_member_fn
@al.jit
def tanh(x: al.array) -> al.array:
    """Tanh."""
    return al.prim.tanh(x)

@_tensor_member_fn
@al.jit
def neg(x: al.array) -> al.array:
    """Negation."""
    return al.prim.neg(x)

@_tensor_member_fn
@al.jit
def div(x: al.array, y: al.array) -> al.array:
    """Division."""
    return al.prim.div(x, y)

