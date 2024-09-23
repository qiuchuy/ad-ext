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
    "u just need to define magic func __setattr__"
    _modules: Dict[str, Optional["Module"]]

    # _parameters: Dict[str, Optional[Parameter]]
    """[TODO]"""
    _buffers: Dict[str, Optional[al.array]]

    """can be called like a function """

    def __init__(self):
        self._no_grad = set()
        super().__setattr__("_training", False)
        # 避免调用 子类的__setattr__
        super().__setattr__("_modules", {})

    def __setattr__(self, key, value):
        modules = self.__dict__.get("_modules")
        if isinstance(value, al.nn.layers.base.Module):
            modules[type(self).__name__ + "." + key] = value
            super().__setattr__(key, value)

        else:
            super().__setattr__(key, value)

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, mode: bool):
        self._training = mode

    def eval(self):
        return self.train(False)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def children(self) -> Iterator["Module"]:
        r"""Return an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        r"""Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

        Yields:
            (str, Module): Tuple containing a name and child module

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module


# def _extra_repr(self):
#     return
# def __repr__(self):
#     print("[TODO]")
#     return

# ====params 部分实现===


# if __name__ == "__main__":
#     import ailang as al
#     import inspect

#     class X(Module):
#         def __init__(self):
#             super().__init__()
#             self.bn = al.nn.Batchnorm2d(3)
#             self.cov = al.nn.Conv2d(1, 2)

#     class Y(Module):
#         def __init__(self):
#             super().__init__()
#             self.bn = al.nn.Batchnorm2d(3)
#             self.cov = al.nn.Conv2d(1, 2)

#     x = X()
#     x.eval()
#     for n, m in x.named_children():
#         print("???", n, m)
#     print(x.training)
