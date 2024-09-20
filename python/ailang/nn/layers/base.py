import ailang as al
import textwrap
from typing import Any, Callable, List, Optional, Tuple, Union, Dict, Iterator, Set
from ailang import array


class Module:
    """base class for building nn with ailang.
    your module should also subcalss this class.
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    """
    __call__: Callable
    _version: int = 1
    _training: bool
    # _parameters: Dict[str, Optional[Parameter]]
    """[TODO]"""
    _buffers: Dict[str, Optional[al.array]]

    """can be called like a function """

    def __init__(self):
        self._no_grad = set()
        self._training = False

    @property
    def training(self): 
        return self._training

    @training.setter
    def training(self, mode: bool):
        self._training = mode

    def eval(self):
        self.training = False  # 调用 setter，将 _training 设为 False

    def train(self):
        self.training = True  # 调用 setter，将 _training 设为 True

    # def _extra_repr(self):
    #     return
    # def __repr__(self):
    #     print("[TODO]")
    #     return

    # ====params 部分实现===