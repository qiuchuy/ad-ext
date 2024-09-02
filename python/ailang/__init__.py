from . import prim
from ._C.libailang import *
from .transform import jit, grad
from .random import randn
from .standard import (
    mean,
)