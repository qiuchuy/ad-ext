import numpy as np
import ailang as al


def g(x, y):
    b = al.transpose(y)
    #a = al.transpose(b)
    #z = al.transpose(a)
    return al.matmul(x, b)

a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 2], [3, 4]])
c = al.from_numpy(a)
d = al.from_numpy(b)
module = al.jit_impl(g, (c, d, ), target="mlir")
print(module)

from iree import compiler as ireec
from iree import runtime as ireert

# Compile using the vmvx (reference) target:
compiled_flatbuffer = ireec.tools.compile_str(
    module,
    input_type="stablehlo",
    target_backends=["vmvx"])

# Register the module with a runtime context.
# Use the "local-task" CPU driver, which can load the vmvx executable:
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled_flatbuffer)
ctx.add_vm_module(vm_module)

# Invoke the function and print the result.
print("INVOKE simple_mul")
arg0 = np.array([[1, 2], [3, 4]], dtype=np.float32)
arg1 = np.array([[5, 6], [7, 8]], dtype=np.float32)
f = ctx.modules.main["g"]
results = f(arg0, arg1).to_host()
print("Results:", results)

