import ailang as al
import math
from ailang.nn.layers.base import Module
from typing import Union
import numpy as np


class BatchNorm(Module):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = al.from_numpy(np.array(eps))
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = al.ones((num_features,))
            self.bias = al.zeros((num_features,))

    def _extra_repr(self):
        return (
            f"{self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={'weight' in self}, "
            f"track_running_stats={self.track_running_stats}"
        )

    def _calc_stats(self, x):
        """
        Calculate the mean and variance of the input tensor across the batch
        and spatial dimensions.

        Args:
            x (array): Input tensor.

        Returns:
            tuple: Tuple containing mean and variance.
        """

        mean = al.mean(x, False)
        var = al.var(x, False)

        return mean, var

    def __call__(self, x):
        mean, var = self._calc_stats(x)
        out = al.Sub(x, mean)

        out = al.Multiply(out, al.rsqrt(al.Add(var, self.eps)))
        return out
