#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/ViewOpGraph.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <string>

#include "ailang/Transforms/Visualize.h"

namespace ainl::ir {
void visualizeModule(mlir::ModuleOp module, const std::string &filename,
                     mlir::MLIRContext &context) {
  // Set up the pass manager
  std::error_code ec;
  llvm::raw_fd_ostream output(filename, ec);
  if (ec) {
    llvm::errs() << "Error opening file: " << ec.message() << "\n";
  }
  mlir::PassManager passManager(&context);
  passManager.addPass(mlir::createPrintOpGraphPass(output));
  // Run the pass
  if (failed(passManager.run(module))) {
    llvm::errs() << "Pass manager failed.\n";
  }
}

} // namespace ainl::ir