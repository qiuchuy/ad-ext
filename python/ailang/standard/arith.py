import ailang as al

from typing import List
from .common import _tensor_member_fn, element_wise


@_tensor_member_fn
@al.jit
@element_wise
def add(x: al.array, y: al.array) -> al.array:
    """Add two tensors."""
    return al.prim.add(x, y)


@_tensor_member_fn
@al.jit
def sqrt(x: al.array) -> al.array:
    return al.prim.sqrt(x)


@_tensor_member_fn
@al.jit
def conv2d(
    input,
    weight,
    window_stride,
    lhs_dilation,
    rhs_dilation,
    padding_args,
    window_reversal,
) -> al.array:
    "Compute the maxpool2d of the input"
    return al.prim.conv2d(
        input,
        weight,
        window_stride,
        lhs_dilation,
        rhs_dilation,
        padding_args,
        window_reversal,
    )


@_tensor_member_fn
@al.jit
def relu(x: al.array) -> al.array:
    """ReLU."""
    return al.prim.relu(x)


@_tensor_member_fn
@al.jit
def batchnorm2d(
    input: al.array,
    scale: al.array,
    offset: al.array,
    mean: al.array,
    variance: al.array,
) -> al.array:
    "Compute the batchnrom of the input with mean,variance"
    """
    :params: input
    :params: scale
    :params: offset
    :params: mean
    :params: variance
    """
    return al.prim.batchnorm2d(input, scale, offset, mean, variance)


@_tensor_member_fn
@al.jit
def maxpool2d(
    x: al.array,
    window_dimensions,
    window_strides,
    base_dilations,
    window_dilations,
    padding,
) -> al.array:
    """
    Maxpool2d.
    params.
    example.
    input = np.array([[[[ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19],
         [20, 21, 22, 23, 24]],
        [[25, 26, 27, 28, 29],
         [30, 31, 32, 33, 34],
         [35, 36, 37, 38, 39],
         [40, 41, 42, 43, 44],
         [45, 46, 47, 48, 49]],
        [[50, 51, 52, 53, 54],
         [55, 56, 57, 58, 59],
         [60, 61, 62, 63, 64],
         [65, 66, 67, 68, 69],
         [70, 71, 72, 73, 74]]]], dtype=np.float32)
        which is a aray with dimensions(1,3,5,5),each dim means batch_size,channels, height,weight
    window_dimensions.  (1,1,3,3) each batch_size, channel has one kerenl with kernel_size = 3.
    window_strides.(1,1,2,2)。because the input's channels is 3, we move 1 each time.
    base_dilation.(1,1,1,1).
    window_dilation.(1,1,1,1)。
    padding_args.(4,2) with initializer{0,0,0,0,0,0,0,0}. which's dim should be (rank(input),2).In every dim,
        will have h,w direction's padding.
    """
    return al.prim.maxpool2d(
        x, window_dimensions, window_strides, base_dilations, window_dilations, padding
    )


@_tensor_member_fn
@al.jit
def avgpool2d(
    x: al.array,
    window_dimensions,
    window_strides,
    base_dilations,
    window_dilations,
    padding,
) -> al.array:
    return al.prim.avgpool2d(
        x, window_dimensions, window_strides, base_dilations, window_dilations, padding
    )


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
@element_wise
def div(x: al.array, y: al.array) -> al.array:
    """Division."""
    return al.prim.div(x, y)


@_tensor_member_fn
@al.jit
@element_wise
def sub(x: al.array, y: al.array) -> al.array:
    """Subtraction."""
    return al.prim.add(x, al.prim.neg(y))


@_tensor_member_fn
@al.jit
@element_wise
def mul(x: al.array, y: al.array) -> al.array:
    """Multiplication."""
    return al.prim.mul(x, y)


@_tensor_member_fn
@al.jit
def matmul(x: al.array, y: al.array) -> al.array:
    """Matrix multiplication."""
    return al.prim.matmul(x, y)
