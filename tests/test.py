# import numpy as np

# from iree import runtime as ireert
# from iree.compiler import compile_str
# # Compile a module.
# SIMPLE_MUL_ASM = """
#   module @arithmetic {
#     func.func @simple_mul(%arg0: tensor<3x2xi64>, %arg1: tensor<1x2xi64>) ->tensor<4x2xi64> {
#       %input0 = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
#       %input1 = stablehlo.constant dense<[[7, 8]]> : tensor<1x2xi64>
#       %0 = stablehlo.concatenate %input0, %input1, dim = 0 : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
#       return %0 :  tensor<4x2xi64>
#     } 
#   }
# """

# # Compile using the vmvx (reference) target:
# compiled_flatbuffer = compile_str(SIMPLE_MUL_ASM, target_backends=["vmvx"])
# # Register the module with a runtime context.
# # Use the "local-task" CPU driver, which can load the vmvx executable:
# config = ireert.Config("local-task")
# ctx = ireert.SystemContext(config=config)
# vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, compiled_flatbuffer)
# ctx.add_vm_module(vm_module)

# # Invoke the function and print the result.
# print("INVOKE simple_mul")
# arg0 = np.ones((3,2), dtype=np.int64)
# arg1 =  np.ones((1,2), dtype=np.int64)
# f = ctx.modules.arithmetic["simple_mul"]
# results = f(arg0, arg1).to_host()
# print("Results:", results)
import numpy as np

from iree import runtime as ireert
from iree.compiler import compile_str
from iree import compiler as ireec

INPUT_MLIR = """
module @maxpool2d {
  func.func @maxpool2d(%arg0: tensor<4x4xf32>) -> tensor<2x2xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 2, 2>, window_strides = array<i64: 2, 2>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<4x4xf32>, tensor<f32>) -> tensor<2x2xf32>
    stablehlo.return %0 : tensor<2x2xf32>
  }
}
"""
compiled_flatbuffer = compile_str(INPUT_MLIR, target_backends=["vmvx"])
# Register the module with a runtime context.
# Use the "local-task" CPU driver, which can load the vmvx executable:
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, compiled_flatbuffer)
ctx.add_vm_module(vm_module)

# Invoke the function and print the result.
print("INVOKE simple_mul")
arg0 = np.array([1., 2., 3., 4.], dtype=np.float32)
arg1 = np.array([4., 5., 6., 7.], dtype=np.float32)
f = ctx.modules.maxpool2d["maxpool2d"]
results = f().to_host()
print("Results:", results)