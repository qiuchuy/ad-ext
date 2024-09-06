#include "ailang/IR/Node.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "ailang/IR/Tensor.h"
#include "ailang/IR/Type.h"
#include "ailang/IR/Value.h"
#include "ailang/Transforms/StablehloConversion.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"

using namespace llvm;
using namespace mlir;

namespace ainl::ir {

StableHLOLoweringPass::StableHLOLoweringPass(mlir::MLIRContext &context,
                                             const std::string &name)
    : builder(&context) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc(), name);
}

mlir::ModuleOp StableHLOLoweringPass::module() { return theModule; }

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

  auto returnType = module->getReturnType();
  if (auto tuple = asType<ir::TupleType>(returnType)) {
    for (auto type : tuple->getTypes()) {
      returnTypes.push_back(
          createRankedTensorTypeFromTensorType(type, *theModule->getContext()));
    }
  } else {
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
  theModule.push_back(function);

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

  if (failed(theModule.verify())) {
    theModule.emitError("module verification error");
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
  builder.create<mlir::stablehlo::ReturnOp>(builder.getUnknownLoc(), value);
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

void StableHLOLoweringPass::visit(AddPtr node) {
  llvm::SmallVector<mlir::Value, 4> inputs = {valueMap[node->getLHS()],
                                              valueMap[node->getRHS()]};
  llvm::SmallVector<mlir::NamedAttribute, 4> attributes;
  auto op = builder.create<mlir::stablehlo::AddOp>(builder.getUnknownLoc(),
                                                   inputs, attributes);
  insertValueMapping(node, op);
}

void StableHLOLoweringPass::visit(ConvolutionPtr node) {
  mlir::Value input = valueMap[node->getInputValue()];
  mlir::Value weight = valueMap[node->getWeightValue()];
  std::initializer_list<int64_t> padding_args = {0, 0, 0, 0};
  auto paddingTensorType =
      mlir::RankedTensorType::get({2, 2}, builder.getIntegerType(64));
  mlir::DenseIntElementsAttr padding =
      mlir::DenseIntElementsAttr::get(paddingTensorType, padding_args);
  auto resultType =
      mlir::RankedTensorType::get({1, 2, 2, 1}, builder.getF32Type());
  llvm::SmallVector<mlir::Type, 1> resultTypes = {resultType};
  auto convArgs = node->getArgs();
  auto windowStrides = builder.getDenseI64ArrayAttr(convArgs[0]);
  auto lhsDilation = builder.getDenseI64ArrayAttr(convArgs[1]); // lhs_dilation
  auto rhsDilation = builder.getDenseI64ArrayAttr(convArgs[2]); // rhs_dilation
  // default 0 0
  auto window_reversal =
      builder.getDenseBoolArrayAttr({0, 0}); // window_reversal

  int64_t inputBatchDimension = 0;
  int64_t inputFeatureDimension = 3;
  llvm::SmallVector<int64_t, 2> inputSpatialDimensions = {1, 2};

  int64_t kernelInputFeatureDimension = 2;
  int64_t kernelOutputFeatureDimension = 3;
  llvm::SmallVector<int64_t, 2> kernelSpatialDimensions = {0, 1};

  int64_t outputBatchDimension = 0;
  int64_t outputFeatureDimension = 3;
  llvm::SmallVector<int64_t, 2> outputSpatialDimensions = {1, 2};

  //  ConvDimensionNumbersAttr
  auto conv_dimension_numbers_attr =
      mlir::stablehlo::ConvDimensionNumbersAttr::get(
          builder.getContext(), inputBatchDimension, inputFeatureDimension,
          inputSpatialDimensions, kernelInputFeatureDimension,
          kernelOutputFeatureDimension, kernelSpatialDimensions,
          outputBatchDimension, outputFeatureDimension,
          outputSpatialDimensions);

  auto precision_config = mlir::ArrayAttr::get(
      builder.getContext(),
      {mlir::stablehlo::PrecisionAttr::get(builder.getContext(),
                                           mlir::stablehlo::Precision::DEFAULT),
       mlir::stablehlo::PrecisionAttr::get(
           builder.getContext(),
           mlir::stablehlo::Precision::DEFAULT)}); // precision_config
  auto op = builder.create<mlir::stablehlo::ConvolutionOp>(
      builder.getUnknownLoc(), resultTypes, input, weight, windowStrides,
      padding, lhsDilation, rhsDilation, window_reversal,
      conv_dimension_numbers_attr, 1, 1, precision_config);
  insertValueMapping(node, op);
}
void StableHLOLoweringPass::visit(ReluPtr node) {
  mlir::Value value = valueMap[node->getValue()];

  auto shape = node->getShape();
  std::vector<int64_t> int64Shape(shape.size());
  std::transform(shape.begin(), shape.end(), int64Shape.begin(),
                 [](int val) { return static_cast<int64_t>(val); });

  auto tensorType = mlir::RankedTensorType::get(
      mlir::ArrayRef<int64_t>(int64Shape),
      mlir::FloatType::getF32(builder.getContext()));
  auto getConstValue = [&](double val) {
    return mlir::DenseElementsAttr::get(
        tensorType, builder.getFloatAttr(tensorType.getElementType(), val));
  };
  auto zeroConstant = getConstValue(0);

  mlir::Value zeroValue = builder.create<mlir::stablehlo::ConstantOp>(
      builder.getUnknownLoc(), zeroConstant);
  auto op = builder.create<mlir::stablehlo::MaxOp>(builder.getUnknownLoc(),
                                                   value, zeroValue);

  insertValueMapping(node, op);
}

mlir::Value computeReduce(mlir::Location loc, mlir::TypeRange resultType,
                          mlir::ValueRange operand, mlir::ValueRange zero,
                          mlir::DenseI64ArrayAttr &dimensions,
                          mlir::OpBuilder &builder) {
  auto CurrentBlock = builder.getInsertionBlock();
  llvm::SmallVector<mlir::Value, 1> inputs = {operand};
  llvm::SmallVector<mlir::Value, 1> initValues = {zero};
  auto reduceOp = builder.create<mlir::stablehlo::ReduceOp>(
      builder.getUnknownLoc(), resultType, inputs, initValues, dimensions);
  auto *Body = builder.createBlock(&reduceOp.getBodyRegion());
  mlir::Type ElementType =
      operand[0].getType().cast<mlir::TensorType>().getElementType();
  mlir::Value Element = Body->addArgument(
      mlir::RankedTensorType::get({}, ElementType), builder.getUnknownLoc());
  mlir::Value Accumulator = Body->addArgument(
      mlir::RankedTensorType::get({}, ElementType), builder.getUnknownLoc());
  builder.setInsertionPointToEnd(Body);
  std::vector<mlir::Value> AddArgs = {Element, Accumulator};
  auto AddResult =
      builder.create<mlir::stablehlo::AddOp>(builder.getUnknownLoc(), AddArgs);
  builder.create<mlir::stablehlo::ReturnOp>(builder.getUnknownLoc(),
                                            AddResult.getResult());
  builder.setInsertionPointToEnd(CurrentBlock);
  return reduceOp->getResult(0);
}

void StableHLOLoweringPass::visit(MeanPtr node) {
  auto CurrentBlock = builder.getInsertionBlock();
  auto ResultType = createRankedTensorTypeFromTensorType(
      node->getType(), *theModule.getContext());
  mlir::Value value = valueMap[node->getValue()];
  mlir::Value initValue = builder.create<mlir::stablehlo::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getZeroAttr(ResultType.getElementType()));
  llvm::SmallVector<mlir::Value, 1> inputs = {value};
  llvm::SmallVector<mlir::Value, 1> initValues = {initValue};
  auto dimensions = builder.getDenseI64ArrayAttr(node->getDim());
  auto reduceOp = builder.create<mlir::stablehlo::ReduceOp>(
      builder.getUnknownLoc(), ResultType, inputs, initValues, dimensions);

  auto *Body = builder.createBlock(&reduceOp.getBodyRegion());
  mlir::Type ElementType =
      value.getType().cast<mlir::TensorType>().getElementType();
  mlir::Value Element = Body->addArgument(
      mlir::RankedTensorType::get({}, ElementType), builder.getUnknownLoc());
  mlir::Value Accumulator = Body->addArgument(
      mlir::RankedTensorType::get({}, ElementType), builder.getUnknownLoc());
  builder.setInsertionPointToEnd(Body);
  std::vector<mlir::Value> AddArgs = {Element, Accumulator};
  auto AddResult =
      builder.create<mlir::stablehlo::AddOp>(builder.getUnknownLoc(), AddArgs);
  builder.create<mlir::stablehlo::ReturnOp>(builder.getUnknownLoc(),
                                            AddResult.getResult());
  builder.setInsertionPointToEnd(CurrentBlock);
  std::vector<int> inShape = node->getShape();

  mlir::RankedTensorType tensorType =
      mlir::RankedTensorType::get({}, ElementType);
  mlir::DenseElementsAttr attr;
  if (ElementType.isa<mlir::FloatType>()) {
    float dimSizeValue = 1;
    for (auto dim : inShape) {
      dimSizeValue *= dim;
    }

    // Create a DenseElementsAttr with the dimSize value
    attr = mlir::DenseElementsAttr::get(tensorType,
                                        llvm::ArrayRef<float>{dimSizeValue});
  } else if (ElementType.isa<mlir::IntegerType>()) {
    int64_t dimSizeValue = 1;
    for (auto dim : inShape) {
      dimSizeValue *= dim;
    }

    // Create a DenseElementsAttr with the dimSize value
    attr = mlir::DenseElementsAttr::get(tensorType,
                                        llvm::ArrayRef<int64_t>{dimSizeValue});
  } else {
    throw std::runtime_error("Unsupported data type lowering MeanOp.");
  }

  // Create a ConstantOp with the tensor type and attribute
  auto dimSizeValue = builder.create<mlir::stablehlo::ConstantOp>(
      builder.getUnknownLoc(), tensorType, attr);
  auto meanValue = builder.create<mlir::stablehlo::DivOp>(
      builder.getUnknownLoc(), reduceOp.getResult(0), dimSizeValue);
  insertValueMapping(node, meanValue);
}
void StableHLOLoweringPass::visit(VariancePtr node) {
  auto CurrentBlock = builder.getInsertionBlock();
  auto ResultType = createRankedTensorTypeFromTensorType(
      node->getType(), *theModule.getContext());
  mlir::Value value = valueMap[node->getValue()];
  mlir::Value initValue = builder.create<mlir::stablehlo::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getZeroAttr(ResultType.getElementType()));
  llvm::SmallVector<mlir::Value, 1> inputs = {value};
  llvm::SmallVector<mlir::Value, 1> initValues = {initValue};
  auto dimensions = builder.getDenseI64ArrayAttr(node->getDim());
  auto reduceOp = builder.create<mlir::stablehlo::ReduceOp>(
      builder.getUnknownLoc(), ResultType, inputs, initValues, dimensions);

  auto *Body = builder.createBlock(&reduceOp.getBodyRegion());
  mlir::Type ElementType =
      value.getType().cast<mlir::TensorType>().getElementType();
  mlir::Value Element = Body->addArgument(
      mlir::RankedTensorType::get({}, ElementType), builder.getUnknownLoc());
  mlir::Value Accumulator = Body->addArgument(
      mlir::RankedTensorType::get({}, ElementType), builder.getUnknownLoc());
  builder.setInsertionPointToEnd(Body);
  std::vector<mlir::Value> AddArgs = {Element, Accumulator};
  auto AddResult =
      builder.create<mlir::stablehlo::AddOp>(builder.getUnknownLoc(), AddArgs);
  builder.create<mlir::stablehlo::ReturnOp>(builder.getUnknownLoc(),
                                            AddResult.getResult());
  builder.setInsertionPointToEnd(CurrentBlock);
  std::vector<int> inShape = node->getShape();

  mlir::RankedTensorType tensorType =
      mlir::RankedTensorType::get({}, ElementType);
  mlir::DenseElementsAttr attr;
  mlir::DenseElementsAttr correction_attr;
  if (ElementType.isa<mlir::FloatType>()) {
    float dimSizeValue = 1;

    for (auto dim : inShape) {
      dimSizeValue *= dim;
    }
    float correction = dimSizeValue - 1;
    // Create a DenseElementsAttr with the dimSize value
    attr = mlir::DenseElementsAttr::get(tensorType,
                                        llvm::ArrayRef<float>{dimSizeValue});
    correction_attr = mlir::DenseElementsAttr::get(
        tensorType, llvm::ArrayRef<float>{correction});
  } else {
    throw std::runtime_error("Unsupported data type lowering VarianceOp.");
  }

  // Create a ConstantOp with the tensor type and attribute
  auto dimSizeValue = builder.create<mlir::stablehlo::ConstantOp>(
      builder.getUnknownLoc(), tensorType, attr);
  auto meanValue = builder.create<mlir::stablehlo::DivOp>(
      builder.getUnknownLoc(), reduceOp.getResult(0), dimSizeValue);

  // Broadcast in dim to match ranks
  std::vector<int64_t> int64Shape(inShape.size());
  std::transform(inShape.begin(), inShape.end(), int64Shape.begin(),
                 [](int val) { return static_cast<int64_t>(val); });
  mlir::Type BroadcastType = mlir::RankedTensorType::get(
      mlir::ArrayRef<int64_t>(int64Shape),
      mlir::FloatType::getF32(builder.getContext()));
  std::vector<int64_t> axes(inShape.size());
  std::iota(axes.begin(), axes.end(), 0);
  mlir::DenseI64ArrayAttr axes_attr = builder.getDenseI64ArrayAttr({});

  auto broadcastOP = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      builder.getUnknownLoc(), BroadcastType, meanValue, axes_attr);
  // Compute substract
  auto subValue = builder.create<mlir::stablehlo::SubtractOp>(
      builder.getUnknownLoc(), value, broadcastOP);
  auto squareValue = builder.create<mlir::stablehlo::MulOp>(
      builder.getUnknownLoc(), subValue, subValue);
  llvm::SmallVector<mlir::Value, 1> squareInputs = {squareValue};
  auto squareSumValue =
      computeReduce(builder.getUnknownLoc(), ResultType, squareInputs,
                    initValues, dimensions, builder);
  auto correctionValue = builder.create<mlir::stablehlo::ConstantOp>(
      builder.getUnknownLoc(), tensorType, correction_attr);
  auto varValue = builder.create<mlir::stablehlo::DivOp>(
      builder.getUnknownLoc(), squareSumValue, correctionValue);
  insertValueMapping(node, varValue);
}

void StableHLOLoweringPass::visit(BatchNorm2dPtr node) {
  // input operand scale offset
  mlir::Value value = valueMap[node->getValue()];
  mlir::Value scale = valueMap[node->getScale()];
  mlir::Value mean = valueMap[node->getMean()];
  mlir::Value offset = valueMap[node->getOffset()];
  mlir::Value variance = valueMap[node->getVariance()];

  // // same shape
  auto resultType = value.getType();
  llvm::SmallVector<mlir::Type, 1> resultTypes = {resultType};
  mlir::FloatAttr epsilon = builder.getF32FloatAttr(0.0);
  mlir::IntegerAttr feature_index = builder.getI64IntegerAttr(2);

  auto op = builder.create<mlir::stablehlo::BatchNormInferenceOp>(
      builder.getUnknownLoc(), resultTypes, value, scale, offset, mean,
      variance, epsilon, feature_index);
  insertValueMapping(node, op);
}

void StableHLOLoweringPass::visit(Maxpool2dPtr node) {
  auto CurrentBlock = builder.getInsertionBlock();
  mlir::Value value = valueMap[node->getValue()];
  auto window_dimensions = builder.getDenseI64ArrayAttr({2, 2});
  auto window_strides = builder.getDenseI64ArrayAttr({2, 2});
  auto base_dilations = builder.getDenseI64ArrayAttr({1, 1});
  auto window_dilations = builder.getDenseI64ArrayAttr({1, 1});
  std::initializer_list<int64_t> padding_args = {0, 0, 0, 0};
  auto paddingTensorType =
      mlir::RankedTensorType::get({2, 2}, builder.getIntegerType(64));
  mlir::DenseIntElementsAttr padding =
      mlir::DenseIntElementsAttr::get(paddingTensorType, padding_args);
  auto resultType = mlir::RankedTensorType::get({2, 2}, builder.getF32Type());

  mlir::Value initValue = builder.create<mlir::stablehlo::ConstantOp>(
      builder.getUnknownLoc(), builder.getZeroAttr(builder.getF32Type()));
  auto ReduceWindowOp = builder.create<mlir::stablehlo::ReduceWindowOp>(
      builder.getUnknownLoc(), resultType, value, initValue, window_dimensions,
      window_strides, base_dilations, window_dilations, padding);
  auto *Body = builder.createBlock(&ReduceWindowOp.getBodyRegion());

  mlir::Type ElementType =
      value.getType().cast<mlir::TensorType>().getElementType();
  mlir::Value Element = Body->addArgument(
      mlir::RankedTensorType::get({}, ElementType), builder.getUnknownLoc());
  mlir::Value Accumulator = Body->addArgument(
      mlir::RankedTensorType::get({}, ElementType), builder.getUnknownLoc());
  builder.setInsertionPointToEnd(Body);
  std::vector<mlir::Value> AddArgs = {Element, Accumulator};
  auto AddResult =
      builder.create<mlir::stablehlo::MaxOp>(builder.getUnknownLoc(), AddArgs);
  builder.create<mlir::stablehlo::ReturnOp>(builder.getUnknownLoc(),
                                            AddResult.getResult());
  builder.setInsertionPointToEnd(CurrentBlock);
  insertValueMapping(node, ReduceWindowOp->getResult(0));
}

void StableHLOLoweringPass::visit(CompareOpPtr node) {
  auto lhs = valueMap[node->getLHS()];
  auto rhs = valueMap[node->getRHS()];
  auto direction =
      mlir::stablehlo::ComparisonDirection(node->getCompareDirection());
  mlir::stablehlo::ComparisonType compareType;
  if (auto tensorType = asType<TensorType>(node->getLHS()->getType())) {
    auto elementType = tensorType->getElementType();
    if (elementType->isFloatType())
      compareType = mlir::stablehlo::ComparisonType::FLOAT;
  }
  compareType = mlir::stablehlo::ComparisonType::SIGNED;
  auto op = builder.create<mlir::stablehlo::CompareOp>(
      builder.getUnknownLoc(), lhs, rhs, direction, compareType);
  insertValueMapping(node, op);
}

mlir::RankedTensorType
createRankedTensorTypeFromTensorType(TypePtr type, mlir::MLIRContext &context) {

  if (auto tensorType = std::dynamic_pointer_cast<TensorType>(type)) {
    // may need to change the data type used in `getConcreteShape` to
    // avoid this
    std::vector<int64_t> shape;
    shape.reserve(tensorType->getConcreteShape().size());

    for (const auto &value : tensorType->getConcreteShape()) {
      shape.push_back(static_cast<int64_t>(value));
    }
    auto elementType =
        createTypeFromElementType(tensorType->getElementType(), context);
    return mlir::RankedTensorType::get(shape, elementType);
  } else if (auto literal = asType<LiteralType>(type)) {
    auto elementType =
        createTypeFromElementType(literal->getValue()->getType(), context);
    return mlir::RankedTensorType::get({}, elementType);
  } else {
    return mlir::RankedTensorType::get(
        {}, createTypeFromElementType(type, context));
  }
}

mlir::Type createTypeFromElementType(TypePtr type, mlir::MLIRContext &context) {
  switch (type->kind()) {
  case Type::TypeKind::BoolType:
    return mlir::IntegerType::get(&context, 1);
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
  return mlirModuleToString(loweringPass->module());
}

std::string mlirModuleToString(mlir::ModuleOp module) {
  std::string str;
  llvm::raw_string_ostream os(str);
  module.print(os);
  return os.str();
}

} // namespace ainl::ir
