#include "ir/type.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
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
    inputTypes.push_back(createRankedTensorTypeFromTensorType(
        paramType, *theModule->getContext()));
  }

  for (auto returnType : module->getReturnTypes()) {
    returnTypes.push_back(createRankedTensorTypeFromTensorType(
        returnType, *theModule->getContext()));
  }

  auto funcType =
      mlir::FunctionType::get(theModule->getContext(), inputTypes, returnTypes);

  llvm::ArrayRef<mlir::NamedAttribute> attrs;
  auto function =
      mlir::func::FuncOp::create(mlir::UnknownLoc::get(theModule->getContext()),
                                 module->getName(), funcType, attrs);
  function.setVisibility(mlir::func::FuncOp::Visibility::Public);
  theModule->push_back(function);

  return function;
}

void StableHLOLoweringPass::run(ModulePtr module) {
  auto graph = module->getGraph();
  auto function = createFunctionOpFromModule(module);

  bool entry = true;
  for (auto block : *graph) {
    if (entry) {
      entry = false;
      builder.setInsertionPointToEnd(function.addEntryBlock());
    } else {
      builder.setInsertionPointToEnd(function.addBlock());
    }
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

void StableHLOLoweringPass::visit(ParamPtr node) {
  mlir::Block *block = builder.getInsertionBlock();
  llvm::SmallVector<mlir::Value, 16> arguments(block->args_begin(),
                                               block->args_end());
  auto params = node->getParams();
  assert(arguments.size() == params.size());
  for (size_t i = 0; i < params.size(); ++i) {
    insertValueMapping(params[i], arguments[i]);
  }
}

void StableHLOLoweringPass::visit(ReturnOpPtr node) {
  auto value = valueMap[node->getReturnValue()];
  auto op =
      builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), value);
}

void StableHLOLoweringPass::visit(TransposePtr node) {
  mlir::Value value = valueMap[node->getValue()];
  auto shape = node->getShape();
  std::vector<int64_t> perm;
  for (size_t i = 0; i < shape.size(); ++i) {
    perm.push_back(shape.size() - 1 - i);
  }
  auto op = builder.create<mlir::stablehlo::TransposeOp>(
      builder.getUnknownLoc(), value, perm);
  insertValueMapping(node, op);
}

void StableHLOLoweringPass::visit(MatmulPtr node) {
  llvm::SmallVector<mlir::Value, 4> inputs = {valueMap[node->getLHS()],
                                              valueMap[node->getRHS()]};
  llvm::SmallVector<mlir::NamedAttribute, 4> attributes;
  auto op = builder.create<mlir::stablehlo::MulOp>(builder.getUnknownLoc(),
                                                   inputs, attributes);
  insertValueMapping(node, op);
}

mlir::RankedTensorType
createRankedTensorTypeFromTensorType(TypePtr type, mlir::MLIRContext &context) {

  if (auto tensorType = std::dynamic_pointer_cast<TensorType>(type)) {
    // may need to change the data type used in `getConcreteShape` to avoid this
    std::vector<int64_t> shape;
    shape.reserve(tensorType->getConcreteShape().size());

    for (const auto &value : tensorType->getConcreteShape()) {
      shape.push_back(static_cast<int64_t>(value));
    }
    auto elementType =
        createTypeFromElementType(tensorType->getElementType(), context);
    return mlir::RankedTensorType::get(shape, elementType);
  } else {
    throw std::runtime_error(
        "Unsupported tensor type when lowering to mlir type.");
  }
}

mlir::Type createTypeFromElementType(TypePtr type, mlir::MLIRContext &context) {
  switch (type->kind()) {
  case Type::TypeKind::IntType:
    return mlir::IntegerType::get(&context, 32);
  case Type::TypeKind::FloatType:
    return mlir::FloatType::getF32(&context);
  case Type::TypeKind::DoubleType:
    return mlir::FloatType::getF64(&context);
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
