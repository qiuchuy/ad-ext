import numpy as np
import ailang as al

from enum import Enum
from iree import compiler as ireec
from iree import runtime as ireert
from typing import Union, Tuple, Callable, List

from ailang import tracer


class TransformPipleline:
    def __init__(self):
        self.transforms = []

    def __init__(self, transform):
        self.transforms = [transform]

    def add_transform(self, transform):
        self.transforms.append(transform)

    def apply(self, module):
        for transform in self.transforms:
            module = transform(module)
        self.transforms.clear()
        return module


def transform(module, *args):
    """
    Apply a series of transformations to a module.
    """
    Pipeline = TransformPipleline()
    for arg in args:
        Pipeline.add_transform(arg)
    Pipeline.apply(module)
    return module


dtype_mapping = {
    al.bool: bool,
    # (yuqiuchu) run IREE with float32
    al.f32: np.float32,
    al.f64: np.float64,
}


def check_device(*tracers):
    if all(tracer.device == "cpu" for tracer in tracers):
        return "llvm-cpu", "local-task", "cpu"
    elif all(tracer.device == "gpu" for tracer in tracers):
        return "cuda", "cuda", "gpu"
    else:
        raise ValueError("All tracers must be on the same known device.")


def flatten(*args):
    flattened_arg = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flattened_arg.extend(arg)
        else:
            flattened_arg.append(arg)
    return flattened_arg


def jit(f: Union[Callable]):
    """
    Decorator that performs just-in-time (JIT) compilation of a function.
    """

    def jitted_f(*args, **kwargs):
        flattened_args = flatten(*args)
        tracer_args = [arg for arg in flattened_args if isinstance(arg, tracer)]
        module = al.trace_impl(f, args)
        print(module)
        mlir_str = module.to_mlir()
        print(mlir_str)
        target_backend, ireert_config, device = check_device(*tracer_args)
        compiled_flatbuffer = ireec.tools.compile_str(
            mlir_str,
            input_type="stablehlo",
            target_backends=[target_backend],
            # extra_args=["--mlir-print-ir-before-all"],
            # output_mlir_debuginfo=True,
        )
        config = ireert.Config(ireert_config)
        ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled_flatbuffer)
        ctx.add_vm_module(vm_module)

        numpy_args = [
            np.array(arg.tolist(), dtype=dtype_mapping[arg.dtype])
            for arg in flattened_args
            if isinstance(arg, tracer)
        ]

        _jitted_f = getattr(ctx.modules, f.__name__)[f.__name__]
        result = _jitted_f(*numpy_args)
        if isinstance(result, tuple):
            al_arrays = []
            for res in result:
                al_arrays.append(al.from_numpy(res.to_host(), device=device))
        else:
            al_arrays = al.from_numpy(result.to_host(), device=device)
        return al_arrays

    return jitted_f


def jvp(f: Union[Callable]):
    def grad_f(*args, **kwargs):
        flattened_args = flatten(*args)
        tracer_args = [arg for arg in flattened_args if isinstance(arg, tracer)]
        module = al.trace_impl(f, args)
        print(module)
        al.grad_impl(module)
        print(module)
        mlir_str = module.to_mlir()
        print(mlir_str)
        target_backend, ireert_config, device = check_device(*tracer_args)
        compiled_flatbuffer = ireec.tools.compile_str(
            mlir_str,
            input_type="stablehlo",
            target_backends=[target_backend],
            # extra_args=["--mlir-print-ir-before-all"],
            # output_mlir_debuginfo=True,
        )
        config = ireert.Config(ireert_config)
        ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled_flatbuffer)
        ctx.add_vm_module(vm_module)

        numpy_args = [
            np.array(arg.tolist(), dtype=dtype_mapping[arg.dtype])
            for arg in flattened_args
            if isinstance(arg, tracer)
        ]

        al_arrays = []
        func_result = None
        tangent_result = []
        for turn in range(len(numpy_args)):
            turn_args = []
            numpy_arg_tangents = [
                np.ones_like(arg) if idx == turn else np.zeros_like(arg)
                for idx, arg in enumerate(numpy_args)
            ]
            turn_args.extend(numpy_args)
            turn_args.extend(numpy_arg_tangents)
            _jitted_f = getattr(ctx.modules, f.__name__)[f.__name__]
            result = _jitted_f(*turn_args)
            assert isinstance(result, tuple)
            assert len(result) == 2 
            func_result = al.from_numpy(result[0].to_host(), device=device)
            tangent_result.append(al.from_numpy(result[1].to_host(), device=device))
        al_arrays.append(func_result)
        al_arrays.extend(tangent_result)
        return al_arrays

    return grad_f
