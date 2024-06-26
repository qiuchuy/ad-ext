import inspect
import numpy as np
import ailang as al

from typing import Union, Tuple, Callable
from ailang import Tensor, ModuleNode, array

from iree import compiler as ireec
from iree import runtime as ireert

from .ast_converter import parse_pycallable

dtype_mapping = {
    al.bool: bool,
    # (yuqiuchu) run IREE with float32
    al.f32: np.float32,
}

def jit(debug: bool = False):
    """
    Decorator that performs just-in-time (JIT) compilation of a function.

    Args:
        debug (bool, optional): If True, the function will be compiled using the "ailang" target
            and the compiled module will be printed. If False (default), the function will be
            compiled using the "mlir" target.

    Returns:
        The decorated function, which is the JIT-compiled version of the original function.
    """
    def _jit(f: Union[Callable]):
        def jitted_f(*args, **kwargs):
            if debug:
                module = al.jit_impl(f, args, target="ailang")
                print(module)
                return f(*args, **kwargs)

            module = al.jit_impl(f, args, target="mlir")
            print(module)
            compiled_flatbuffer = ireec.tools.compile_str(
                module,
                input_type="stablehlo",
                target_backends=["vmvx"],
                #extra_args=["--mlir-print-ir-before-all"],
                #output_mlir_debuginfo=True,
            )

            config = ireert.Config("local-task")
            ctx = ireert.SystemContext(config=config)
            vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled_flatbuffer)
            ctx.add_vm_module(vm_module)

            numpy_args = [np.array(arg.tolist(), dtype=dtype_mapping[arg.dtype]) for arg in args]

            _jitted_f = ctx.modules.main[f.__name__]
            results = _jitted_f(*numpy_args).to_host()
            return results

        return jitted_f
    return _jit

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

