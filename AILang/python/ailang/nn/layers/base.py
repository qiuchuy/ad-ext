import ailang as al
import textwrap
from typing import Any, Callable, List, Optional, Tuple, Union, Dict, Iterator, Set
from ailang import tensor


class Module:
    """base class for building nn with ailang.
    your module should also subcalss this class.
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    """

    __call__: Callable
    _version: int = 1
    training: bool
    # _parameters: Dict[str, Optional[Parameter]]
    """[TODO]"""
    _buffers: Dict[str, Optional[al.tensor]]

    """can be called like a function """

    def __init__(self):
        self._no_grad = set()
        self._training = True

    @property
    def training(self):
        return self._training

    # def _extra_repr(self):
    #     return
    # def __repr__(self):
    #     print("[TODO]")
    #     return

    # ====params 部分实现===

    @staticmethod
    def valid_parameter_filter(module, key, value):
        return isinstance(value, (Module, list, al.ffi.libailang.Tensor))

    @staticmethod
    def valid_module_filter(module, key, value):
        return isinstance(value, (dict, list))

    def parameters(self, recurse: bool = True):
        return self.module_filter(self.valid_parameter_filter)

    def module_filter(
        self,
        filter_fn: Callable[["Module", str, Any], bool],
        map_fn: Optional[Callable] = None,
        isleaf_fn: Optional[Callable] = None,
    ):
        map_fn = map_fn or (lambda x: x)
        isleaf_fn = isleaf_fn or (
            lambda m, k, v: not isinstance(v, (Module, dict, list))
        )

        def _unpack_params(vk, v):
            if isleaf_fn(self, vk, v):
                return map_fn(v)

            if isinstance(v, Module):
                next_dict = {}
                for k, v in v.__dict__.items():
                    prefix = f"{vk}.{k}"
                    if filter_fn(self, prefix, v):
                        next_dict[prefix] = _unpack_params(prefix, v)
                return next_dict
            if isinstance(v, list):
                next_list = []
                for i, vi in enumerate(v):
                    prefix = f"{vk}.{i}"
                    next_list.append(
                        _unpack_params(prefix, vi)
                        if filter_fn(self, prefix, vi)
                        else {}
                    )
                return next_list

            raise RuntimeError("Unexpected leaf found while traversing the module")

        return {
            k: _unpack_params(k, v)
            for k, v in self.__dict__.items()
            if filter_fn(self, k, v)
        }

    def _child_modules(self, value):
        return self.module_filter(
            self.valid_module_filter, isleaf_fn=lambda m, k, v: isinstance(v, (Module))
        )

    # weights 部分实现
