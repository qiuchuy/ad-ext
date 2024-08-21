import ailang as al
import math
from ailang.nn.layers.base import Module
from typing import Union

"""Applies a 2-dimensional convolution over the multi-channel input image.

    The channels are expected to be last i.e. the input shape should be ``NHWC`` where:
        - ``N`` is the batch dimension
        - ``H`` is the input image height
        - ``W`` is the input image width
        - ``C`` is the number of input channels
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
"""


class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        kernel_size, stride, padding = map(
            lambda x: (x, x) if isinstance(x, int) else x,
            (kernel_size, stride, padding),
        )
        if bias:
            raise NotImplementedError()
        self.weight = al.tensor((out_channels, *kernel_size, in_channels), "Float")
        self.padding = padding
        self.stride = stride
        # print("type", type(self.weight))

    def __call__(self, x):
        def f_conv(x):
            return al.convolution(x)

        return f_conv
        pass
        y = al.convolution(x)
        if "bias" in self:
            y = y + self.bias
        return y
