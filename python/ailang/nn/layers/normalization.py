import ailang as al
import math
import numpy as np
from ailang.nn.layers.base import Module
from typing import Union, Tuple,Optional


class Batchnorm2d(Module):
    """
    
    """
    def __init__(
        self,
        num_features :int,
        eps:float,
        momentum:Optional[float],
        affine:bool,
        track_running_stats:bool
    ):
        super().__init__()
        assert not affine and not track_running_stats, "[nn.Batchnorm2d] tracking and affine not implement yet."
        self.num_features =   num_features
        self.eps = eps
        
    def compute_mean(x:al.array):
        return al.standard.mean()        
    
