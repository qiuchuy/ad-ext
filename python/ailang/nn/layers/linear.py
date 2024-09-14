import math
from typing import Any, Tuple
import numpy as np
import ailang as al
from ailang.nn.layers.base import Module


class Linear(Module):
    """Applies an affine transformation to the input.

    .. math::

        y = x W^\top + b
    """

    @staticmethod
    def get_random_array(self, shape: Tuple[int], dtype: np.dtype):
        np_array = np.random.randn(*shape).astype(dtype)
        return al.from_numpy(np_array)

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)

        # 实现随机数据支持之前，先
        self.weight = self.get_random_array((output_dims, input_dims), np.float32)
        # self.weight = al.random.uniform(
        #     low=-scale,
        #     high=scale,
        #     shape=(output_dims, input_dims),
        # )
        if bias:
            self.bias = self.get_random_array((output_dims), np.float32)
            # self.bias = al.random.uniform(
            #     low=-scale,
            #     high=scale,
            #     shape=(output_dims,),
            # )

    def _extra_repr(self) -> str:
        return f"input_dims={self.weight.shape[1]}, output_dims={self.weight.shape[0]}, bias={'bias' in self}"

    def __call__(self, x: al.array) -> al.array:
        if "bias" in self:

            x = al.standard.matmul(self.weight, x)
            x = al.standard.add(x, self.bias)
        else:
            x = al.standard.matmul(self.weight, x)
        return x
