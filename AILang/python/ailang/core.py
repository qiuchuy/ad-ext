import inspect
from typing import Union, Tuple, Callable

from ailang import Tensor, ModuleNode, array

from .ast_converter import parse_pycallable


def compile(f: Union[Callable]):
    """
    Compile a python callable into a MLIR Module.
    :param f: Function to be compiled.
    :return: A compiled python callable represented in mlir module
    """
    ast = parse_pycallable(f)

    def compiled_f(*args: Union[Tuple[Tensor], Tensor]):
        ast.type_infer(args)
        mlir_module = ast.to_mlir()
        return mlir_module(*args)

    return compiled_f


def compile_ast(f: Callable, *args: Union[Tuple[Tensor], Tensor]) -> ModuleNode:
    """
    Compile a python callable into a typed AST.
    :param f: Function to be compiled
    :param args: input arguments
    :return: a typed AST ModuleNode
    """
    ast = parse_pycallable(f)
    arg_names = list(inspect.signature(f).parameters.values())
    arg_names = [str(name) for name in arg_names]
    return ast.type_infer(arg_names, *args)


def compile_ir(f: Callable, *args: Union[Tuple[Tensor], Tensor]):
    """
    Compile a python callable into AINL typed IR.
    :param f: function to be compiled
    :param args: input arguments
    :return: typed AINL IR Module
    """
    ast = parse_pycallable(f)
    arg_names = list(inspect.signature(f).parameters.values())
    arg_names = [str(name) for name in arg_names]
    return ast.ir_lowering(arg_names, *args)

