#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"

#include <fstream>
#include <string>

namespace ainl::ir {

void visualizeModule(mlir::ModuleOp module, const std::string &filename,
                     mlir::MLIRContext &context);

} // end namespace ainl::ir