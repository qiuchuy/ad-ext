import ailang as al
import math
import numpy as np
from ailang.nn.layers.base import Module
from typing import Union, Tuple, Optional


class Batchnorm2d(Module):
    """
    in the forward pass, the standard-deviation is calculated via the biased estimator, equivalent to ~torch.var(input, unbiased=False)
    """

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
        self.running_mean = np.zeros((num_features), dtype=np.float32)
        self.running_var = np.ones((num_features), dtype=np.float32)

    def compute_mean(self, x: al.array):
        return al.mean(x, [0, 2, 3])

    def compute_var(self, x: al.array):
        # biased
        return al.var(x, [0, 2, 3], 0)

    def get_running_mean(self):
        assert self.running_mean is not None, "you can get running_mean after inference"
        return self.running_mean

    def get_running_var(self):
        assert self.running_var is not None, "you can get running_var after inference"
        return self.running_var

    def __call__(self, x: al.array):
        np_scale = np.ones((self.num_features), dtype=np.float32)
        np_offset = np.zeros((self.num_features), dtype=np.float32)
        self.offset = al.from_numpy(np_offset, device=x.device)
        self.scale = al.from_numpy(np_scale, device=x.device)
        if self.training:
            self.running_mean = self.compute_mean(x)
            self.running_var = self.compute_var(x)
        else:
            self.running_mean = al.from_numpy(
                np.zeros((self.num_features), dtype=np.float32), device=x.device
            )
            self.running_var = al.from_numpy(
                np.ones((self.num_features), dtype=np.float32), device=x.device
            )
        return al.batchnorm2d(
            x, self.scale, self.offset, self.running_mean, self.running_var
        )
