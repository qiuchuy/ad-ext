from ._C.libailang import *
from ._C.libailang import _register_eval_callback
from .random import randn
from .transform import jit, grad
from . import prim
from . import standard

_register_eval_callback("transpose", standard.transpose)
_register_eval_callback("mean", standard.mean)
_register_eval_callback("relu", standard.relu)