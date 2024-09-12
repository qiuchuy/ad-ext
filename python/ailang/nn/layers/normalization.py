import ailang as al
import math
import numpy as np
from ailang.nn.layers.base import Module
from typing import Union, Tuple, Optional


class Batchnorm2d(Module):
    """ """

    def __init__(
        self,
        num_features: int,
        eps: float = 0.0000001,
        momentum: Optional[float] = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ):
        super().__init__()
        assert (
            not affine and not track_running_stats
        ), "[nn.Batchnorm2d] tracking and affine not implement yet."
        self.num_features = num_features
        self.eps = eps
        np_scale = np.ones((num_features), dtype=np.float32)
        np_offset = np.zeros((num_features), dtype=np.float32)
        self.offset = al.from_numpy(np_offset)
        self.scale = al.from_numpy(np_scale)

    def compute_mean(self, x: al.array):
        return al.standard.mean(x, [0, 2, 3])

    def compute_var(self, x: al.array):
        return al.standard.var(x, [0, 2, 3])

    def __call__(self, x: al.array):
        return al.standard.batchnorm2d(
            x, self.scale, self.offset, self.compute_mean(x), self.compute_var(x)
        )
