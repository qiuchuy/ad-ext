import math
from typing import Any

import ailang as al
from ailang.nn.layers.base import Module


def _make_activation_module(f):
    def decorator(klass):
        klass.__doc__ = f.__doc__
        klass.__call__ = lambda self, x: f(x)
        return klass

    return decorator


def sigmoid(x):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
    """

    return al.sigmoid(x)


def relu(x):
    return al.maximum(x, al.zeros((1,)))
