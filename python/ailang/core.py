from typing import Union, Tuple, Callable

from ailang import Tensor

from .ast_converter import parse_pycallable


def compile(f: Union[Callable]):
    """
    Compile a python callable into a MLIR Module.
    :param f: Function to be compiled.
    :return: A compiled python callable represented in mlir module
    """
    ast = parse_pycallable(f)

    def compiled_f(*args: Union[Tuple[Tensor], Tensor]):
        alir = ast.compile(args)
        mlir_module = alir.to_mlir()
        return mlir_module(*args)

    return compiled_f
