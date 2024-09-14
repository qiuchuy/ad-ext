from ._C.libailang import *
from ._C.libailang import _register_eval_callback
from .random import randn
from .transform import jit, grad
from . import prim
from . import standard
from . import nn
from .prim import add as _add, mul as _mul, div as _div
from .standard import element_wise


@element_wise
def add(x, y):
    return _add(x, y)


@element_wise
def mul(x, y):
    return _mul(x, y)


@element_wise
def div(x, y):
    return _div(x, y)


_register_eval_callback("add", standard.add)
_register_eval_callback("broadcast_to", standard.broadcast_to)
_register_eval_callback("cat", standard.cat)
_register_eval_callback("div", standard.div)
_register_eval_callback("exp", standard.exp)
_register_eval_callback("matmul", standard.matmul)
_register_eval_callback("mul", standard.mul)
_register_eval_callback("neg", standard.neg)
_register_eval_callback("tanh", standard.tanh)
_register_eval_callback("transpose", standard.transpose)
_register_eval_callback("mean", standard.mean)
_register_eval_callback("relu", standard.relu)
_register_eval_callback("var", standard.var)
_register_eval_callback("batchnorm2d", standard.batchnorm2d)
_register_eval_callback("maxpool2d", standard.maxpool2d)
_register_eval_callback("conv2d", standard.conv2d)
