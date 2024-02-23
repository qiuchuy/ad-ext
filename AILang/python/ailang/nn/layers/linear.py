import math
from typing import Any

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

        #实现随机数据支持之前，先
        self.weight = al.random.randn((output_dims, input_dims),'Float')  
        # self.weight = al.random.uniform(
        #     low=-scale,
        #     high=scale,
        #     shape=(output_dims, input_dims),
        # )
        if bias:
            self.bias = al.random.randn((output_dims), 'Float')
            # self.bias = al.random.uniform(
            #     low=-scale,
            #     high=scale,
            #     shape=(output_dims,),
            # )

    def _extra_repr(self) -> str:
        return f"input_dims={self.weight.shape[1]}, output_dims={self.weight.shape[0]}, bias={'bias' in self}"

    def __call__(self, x: al.tensor) -> al.tensor:
        if "bias" in self:
            # 先matmul 后add addmm[TODO]
    
            x = al.matmul(x, self.weight.T)
            x = al.add(x + self.bias)
        else:
            x = x @ self.weight.T
        return x