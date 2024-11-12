from typing import Any
from ailang.nn.layers.base import Module


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = list(modules)

    def __call__(self, x) -> Any:
        for layer in self.layers:
            x = layer(x)
        return x
