#include "ir/type.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"

#include "ir/tensor.h"
#include "pass/stablehlo_lowering.h"

namespace ainl::ir {

StableHLOLoweringPass::StableHLOLoweringPass(mlir::MLIRContext &context,
                                             const std::string &name)
    : builder(&context) {
  *theModule = mlir::ModuleOp::create(builder.getUnknownLoc(), "main");
}

mlir::ModuleOp *StableHLOLoweringPass::module() { return theModule; }

void StableHLOLoweringPass::insertValueMapping(ValuePtr value,
                                               mlir::Value mlirValue) {
  valueMap[value] = mlirValue;
}

mlir::func::FuncOp
StableHLOLoweringPass::createFunctionOpFromModule(ModulePtr module) {
  auto graph = module->getGraph();

  auto params = graph->getParams();
  llvm::SmallVector<mlir::Type, 16> inputTypes;
  llvm::SmallVector<mlir::Type, 4> returnTypes;
  for (auto paramType : module->getParamTypes()) {
    if (auto tensorType = std::dynamic_pointer_cast<TensorType>(paramType)) {
      inputTypes.push_back(createRankedTensorTypeFromTensorType(
          tensorType, *theModule->getContext()));
    } else {
      throw std::runtime_error(
          "Unsupported parameter type when lowering down to mlir type.");
    }
  }

  for (auto returnType : module->getReturnTypes()) {
    if (auto tensorType = std::dynamic_pointer_cast<TensorType>(returnType)) {
      returnTypes.push_back(createRankedTensorTypeFromTensorType(
          tensorType, *theModule->getContext()));
    } else {
      throw std::runtime_error(
          "Unsupported return type when lowering down to mlir type.");
    }
  }

  auto funcType =
      mlir::FunctionType::get(theModule->getContext(), inputTypes, returnTypes);

  llvm::ArrayRef<mlir::NamedAttribute> attrs;
  auto function =
      mlir::func::FuncOp::create(mlir::UnknownLoc::get(theModule->getContext()),
                                 module->getName(), funcType, attrs);
  function.setVisibility(mlir::func::FuncOp::Visibility::Public);
  theModule->push_back(function);

  mlir::Block *block = function.addEntryBlock();
  builder.setInsertionPointToEnd(block);
  llvm::SmallVector<mlir::Value, 16> arguments(block->args_begin(),
                                               block->args_end());
  assert(arguments.size() == params.size());
  for (size_t i = 0; i < params.size(); ++i) {
    insertValueMapping(params[i], arguments[i]);
  }

  return function;
}

void StableHLOLoweringPass::run(ModulePtr module) {
  LOG_DEBUG("%s", "[pass] Start running StableHLO lowering pass");
  auto graph = module->getGraph();
  auto function = createFunctionOpFromModule(module);
  LOG_DEBUG("%s", "[pass] Successfully create mlir function op from module");
  for (auto block : *graph) {
    LOG_DEBUG("%s", "[pass] Start lowering a new block");
    // auto currentBlock = function.addBlock();
    // builder.setInsertionPointToEnd(currentBlock);
    for (auto node : *block) {
      LOG_DEBUG("[pass] Lowering [%s] down to stablehlo",
                std::string(*node).c_str());
      node->accept(this);
    }
  }
  if (failed(theModule->verify())) {
    theModule->emitError("module verification error");
    return;
  }
}

void StableHLOLoweringPass::visit(NodePtr node) {}

void StableHLOLoweringPass::visit(ParamPtr node) {}

void StableHLOLoweringPass::visit(ReturnOpPtr node) {
  auto loc = builder.getUnknownLoc();
  auto value = valueMap[node->getReturnValue()];
  auto op = builder.create<mlir::func::ReturnOp>(loc, value);
  // insertValueMapping(node, op);
}

void StableHLOLoweringPass::visit(TransposePtr node) {}

void StableHLOLoweringPass::visit(MatmulPtr node) {
  llvm::SmallVector<mlir::Value, 4> inputs = {valueMap[node->getLHS()],
                                              valueMap[node->getRHS()]};
  auto loc = builder.getUnknownLoc();
  llvm::SmallVector<mlir::NamedAttribute, 4> attributes;
  auto op = builder.create<mlir::stablehlo::MulOp>(loc, inputs, attributes);
  insertValueMapping(node, op);
}

mlir::RankedTensorType
createRankedTensorTypeFromTensorType(TensorTypePtr type,
                                     mlir::MLIRContext &context) {
  // may need to change the data type used in `getConcreteShape` to avoid this
  std::vector<int64_t> shape;
  shape.reserve(type->getConcreteShape().size());

  for (const auto &value : type->getConcreteShape()) {
    shape.push_back(static_cast<int64_t>(value));
  }
  auto elementType = createTypeFromElementType(type->getElementType(), context);
  return mlir::RankedTensorType::get(shape, elementType);
}

mlir::Type createTypeFromElementType(TypePtr type, mlir::MLIRContext &context) {
  switch (type->kind()) {
  case Type::TypeKind::IntType:
    return mlir::IntegerType::get(&context, 32);
  case Type::TypeKind::FloatType:
    return mlir::FloatType::getF32(&context);
  default:
    throw std::runtime_error(
        "Unsupported element type when lowering to mlir type.");
  }
}

std::string StableHLOLowering(ModulePtr module) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  auto loweringPass =
      std::make_unique<StableHLOLoweringPass>(context, module->getName());
  loweringPass->run(module);
  return mlirModuleToString(*loweringPass->module());
}

std::string mlirModuleToString(mlir::ModuleOp module) {
  std::string str;
  llvm::raw_string_ostream os(str);
  module.print(os);
  return os.str();
}

} // namespace ainl::ir
