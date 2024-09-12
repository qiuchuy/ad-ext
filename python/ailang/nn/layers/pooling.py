import ailang as al
import math
import numpy as np
from ailang.nn.layers.base import Module
from typing import Union, Tuple


class Maxpool2d(Module):
    def __init__(
        self,
        kernel_size: Union[int, tuple] = 3,
        stride: Union[int, tuple] = 1,
        base_dilations: Union[int, tuple] = 1,
        window_dilations: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
    ):
        super().__init__()
        kernel_size, stride, base_dilations, window_dilations = map(
            lambda x: (1, 1, x, x) if isinstance(x, int) else x,
            (kernel_size, stride, base_dilations, window_dilations),
        )
        (padding,) = map(
            lambda x: (x, x, x, x, x, x, x, x) if isinstance(x, int) else x, (padding,)
        )
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.base_dilations = base_dilations
        self.window_dilations = window_dilations

    def get_random_array(self, shape: Tuple[int], dtype: np.dtype):
        np_array = np.random.randn(*shape).astype(dtype)
        return al.from_numpy(np_array)

    def __call__(self, input):
        assert len(input.shape) == 4,"[nn.Maxpool2d] input's dims expected to be (N,C,H,W)"
        y = al.standard.maxpool2d(
            input,
            self.kernel_size,
            self.stride,
            self.base_dilations,
            self.window_dilations,
            self.padding,
        )
        return y
