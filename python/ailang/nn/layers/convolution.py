import ailang as al
import math
import numpy as np
from ailang.nn.layers.base import Module
from typing import Union, Tuple


class Conv2d(Module):
    """Applies a 2-dimensional convolution over the multi-channel input image.

    The channels are expected to be last i.e. the input shape should be ``NCHW`` where:
        - ``N`` is the batch dimension
        - ``C`` is the number of input channels
        - ``H`` is the input image height
        - ``W`` is the input image width
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int or tuple): The size of the convolution filters.
        stride (int or tuple, optional): The size of the stride when
            applying the filter. Default: 1.
        padding (int or tuple, optional): How many positions to 0-pad
            the input with. Default: 0.
        bias (bool, optional): If ``True`` add a learnable bias to the
            output. Default: ``True``
    Dimensions:
        input: dimensions are expected to be (N, C, H, W); Resluts have the same Shape.
        weight: dimensions are expected to be (C, H, W, O); where 'O' is 'outChannels'.
        TBF:
        in conv, window_dilation mean base img, but kernel_dilation means conv kernel.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride: Union[int, tuple] = 1,
        kernel_size: Union[int, tuple] = 3,
        base_dilation: Union[int, tuple] = 1,
        kernel_dilation: Union[int, tuple] = 1,
        window_reversal: Union[int, tuple] = 0,
        padding: Union[int, tuple] = 0,
        bias: bool = False,
    ):
        super().__init__()
        kernel_size, stride, base_dilation, kernel_dilation, window_reversal = map(
            lambda x: (x, x) if isinstance(x, int) else x,
            (kernel_size, stride, base_dilation, kernel_dilation, window_reversal),
        )
        # must be iterable
        (padding,) = map(
            lambda x: (x, x, x, x) if isinstance(x, int) else x, (padding,)
        )
        assert (
            window_reversal[0] == 0 and window_reversal[0] == 0
        ), "[nn.Conv2d] window_reverse not implement yet."
        if bias:
            raise NotImplementedError()
        self.weight = self.get_random_array(
            (out_channels, in_channels, *kernel_size), np.float32
        )
        self.padding = padding
        self.stride = stride
        self.base_dilation = base_dilation
        self.kernel_dilation = kernel_dilation
        self.window_reversal = window_reversal
        # print("type", type(self.weight))

    def get_random_array(self, shape: Tuple[int], dtype: np.dtype):
        np_array = np.random.randn(*shape).astype(dtype)
        return al.from_numpy(np_array)

    def __call__(self, input):

        y = al.conv2d(
            input,
            self.weight,
            self.stride,
            self.base_dilation,
            self.kernel_dilation,
            self.padding,
            self.window_reversal,
        )
        # if "bias" in self:
        #     y = y + self.bias
        return y
