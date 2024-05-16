import math
from typing import Any

import numpy as np
import ailang as al
from ailang.nn.layers.base import Module


class Linear(Module):
    """Applies an affine transformation to the input.

    .. math::

        y = x W^\top + b

    """

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)

        # 实现随机数据支持之前，先
        self.weight = al.from_numpy(
            np.random.uniform(-scale, scale, (output_dims, input_dims))
        )

        if bias:
            self.bias = self.weight = al.from_numpy(
                np.random.uniform(-scale, scale, (output_dims))
            )

    def _extra_repr(self) -> str:
        return f"input_dims={self.weight.shape[1]}, output_dims={self.weight.shape[0]}, bias={'bias' in self}"

    def __call__(self, x: al.tensor) -> al.tensor:
        x = al.Multiply(self.weight, x)
        x = al.Add(x, self.bias)
        return x
