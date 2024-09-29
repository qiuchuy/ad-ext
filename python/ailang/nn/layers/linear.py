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

    def gen_random_nparray(self, shape, dtype: np.dtype) -> np.ndarray:
        if len(shape):
            random_nparray = np.random.randn(*shape).astype(dtype)
            return random_nparray
        else:
            return dtype(np.random.randn())

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)

        # 实现随机数据支持之前，先
        self.weight = al.from_numpy(
            self.gen_random_nparray((output_dims, input_dims), np.float32)
        )
        # self.weight = al.random.uniform(
        #     low=-scale,
        #     high=scale,
        #     shape=(output_dims, input_dims),
        # )
        if bias:
            self.bias = al.from_numpy(
                self.gen_random_nparray((output_dims,), np.float32)
            )
            # self.bias = al.random.uniform(
            #     low=-scale,
            #     high=scale,
            #     shape=(output_dims,),
            # )

    def _extra_repr(self) -> str:
        return f"input_dims={self.weight.shape[1]}, output_dims={self.weight.shape[0]}, bias={'bias' in self}"

    def __call__(self, x: al.array) -> al.array:
        if "bias" in self.__dict__:
            x = al.matmul(x, al.transpose(self.weight, [1, 0]))
            x = al.add(x, self.bias)
            return x
        else:
            x = al.matmul(x, al.transpose(self.weight, [1, 0]))
        return x
