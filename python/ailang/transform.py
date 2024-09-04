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


def jit(f: Union[Callable]):
    """
    Decorator that performs just-in-time (JIT) compilation of a function.
    """

    def jitted_f(*args, **kwargs):
        tracer_args = [arg for arg in args if isinstance(arg, tracer)]
        tracer_kwargs = [arg for arg in kwargs.values() if isinstance(arg, tracer)]
        tracer_args.extend(tracer_kwargs)
        module = al.trace_impl(f, args)
        print(module)
        mlir_str = module.to_mlir()
        print(mlir_str)
        target_backend, ireert_config, device = check_device(*tracer_args)
        compiled_flatbuffer = ireec.tools.compile_str(
            module.to_mlir(),
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
            np.array(arg.tolist(), dtype=dtype_mapping[arg.dtype]) for arg in tracer_args
        ]

        _jitted_f = getattr(ctx.modules, f.__name__)[f.__name__]
        results = _jitted_f(*numpy_args).to_host()
        al_array = al.from_numpy(results, device=device)
        return al_array

    return jitted_f


def grad(f: Union[Callable]):
    """
    Decorator that performs automatic differentiation of a function.
    """

    def grad_f(*args, **kwargs):
        tracer_args = [arg for arg in args if isinstance(arg, tracer)]
        tracer_kwargs = [arg for arg in kwargs.values() if isinstance(arg, tracer)]
        tracer_args.extend(tracer_kwargs)
        module = al.grad_impl(f, tracer_args)
        print(module)
        mlir_str = module.to_mlir()
        print(mlir_str)
        target_backend, ireert_config, device = check_device(*tracer_args)
        compiled_flatbuffer = ireec.tools.compile_str(
            module.to_mlir(),
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
            np.array(arg.tolist(), dtype=dtype_mapping[arg.dtype]) for arg in args
        ]
        numpy_arg_tangents = [np.ones_like(arg) for arg in numpy_args]

        numpy_args.extend(numpy_arg_tangents)

        _jitted_f = getattr(ctx.modules, f"d{f.__name__}")[f"d{f.__name__}"]
        results = _jitted_f(*numpy_args).to_host()
        return al.from_numpy(results, device=device)

    return grad_f
