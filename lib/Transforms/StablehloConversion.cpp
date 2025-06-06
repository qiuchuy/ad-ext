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
#include <functional>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "ailang/IR/Container.h"
#include "ailang/IR/Tensor.h"
#include "ailang/IR/Type.h"
#include "ailang/IR/Value.h"
#include "ailang/Transforms/StablehloConversion.h"
#include "ailang/Transforms/Visualize.h"
#include "ailang/Transforms/utils.h"
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
  auto Value = valueMap[node->getReturnValue()];
  if (asValueType<TupleContainer>(node->getReturnValue())) {
    auto Tuple = asValueType<TupleContainer>(node->getReturnValue());
    llvm::SmallVector<mlir::Value, 4> Values;
    for (auto &Value : Tuple->getValues()) {
      Values.push_back(valueMap[Value]);
    }
    builder.create<mlir::stablehlo::ReturnOp>(builder.getUnknownLoc(), Values);
  } else {
    builder.create<mlir::stablehlo::ReturnOp>(builder.getUnknownLoc(), Value);
  }
}

void StableHLOLoweringPass::visit(TransposePtr node) {
  mlir::Value value = valueMap[node->getValue()];
  auto shape = node->getShape();
  std::vector<int64_t> perm;
  auto axes = node->getAxes();
  if (axes.size() == 0) {
    for (size_t i = 0; i < shape.size(); i++) {
      perm.push_back(shape.size() - 1 - i);
    }
  } else {
    for (size_t i = 0; i < shape.size(); ++i) {
      perm.push_back(static_cast<int64_t>(axes[i]));
    }
  }
  auto Op = builder.create<mlir::stablehlo::TransposeOp>(
      builder.getUnknownLoc(), value, perm);
  insertValueMapping(node, Op);
}

void StableHLOLoweringPass::visit(ReshapePtr node) {
  mlir::Value value = valueMap[node->getOperand(0)];
  auto Op = builder.create<mlir::stablehlo::ReshapeOp>(
      builder.getUnknownLoc(),
      createRankedTensorTypeFromTensorType(node->getType(),
                                           *theModule->getContext()),
      value);
  insertValueMapping(node, Op);
}

void StableHLOLoweringPass::visit(MulPtr node) {
  mlir::Value lhs = valueMap[node->getOperand(0)];
  mlir::Value rhs = valueMap[node->getOperand(1)];
  auto op = builder.create<mlir::stablehlo::MulOp>(builder.getUnknownLoc(),
                                                   lhs.getType(), lhs, rhs);
  insertValueMapping(node, op);
}

void StableHLOLoweringPass::visit(ConstantDefPtr Node) {
  auto ResultTensorType = asType<TensorType>(Node->getType());
  auto ElementType = ResultTensorType->getElementType();
  if (ElementType->isFloatType()) {
    auto *FloatContainer = asValueType<TupleContainer>(Node->getOperand(0));
    std::vector<float> FloatValues;
    for (auto &Value : FloatContainer->getValues()) {
      auto *LiteralValue = asValueType<Literal>(Value);
      FloatValues.push_back(LiteralValue->getFloatConcreteValue());
    }
    auto ResultType = createRankedTensorTypeFromTensorType(
        ResultTensorType, *theModule.getContext());
    mlir::DenseElementsAttr ConstantAttr =
        mlir::DenseElementsAttr::get(ResultType, ArrayRef<float>(FloatValues));
    auto Op = builder.create<mlir::stablehlo::ConstantOp>(
        builder.getUnknownLoc(), ResultType, ConstantAttr);
    insertValueMapping(Node, Op);
  } else if (ElementType->isIntType()) {
    auto *IntContainer = asValueType<TupleContainer>(Node->getOperand(0));
    std::vector<int> IntValues;
    for (auto &Value : IntContainer->getValues()) {
      auto *LiteralValue = asValueType<Literal>(Value);
      IntValues.push_back(LiteralValue->getIntConcreteValue());
    }
    auto ResultType = createRankedTensorTypeFromTensorType(
        ResultTensorType, *theModule.getContext());
    mlir::DenseElementsAttr ConstantAttr =
        mlir::DenseElementsAttr::get(ResultType, ArrayRef<int>(IntValues));
    auto Op = builder.create<mlir::stablehlo::ConstantOp>(
        builder.getUnknownLoc(), ResultType, ConstantAttr);
    insertValueMapping(Node, Op);
  } else {
    throw std::runtime_error(
        "Unsupported element type when lowering a constant node.");
  }
}

void StableHLOLoweringPass::visit(MatmulPtr node) {
  auto LHSType = asType<TensorType>(node->getLHS()->getType());
  auto RHSType = asType<TensorType>(node->getRHS()->getType());
  auto NodeTensorType = asType<TensorType>(node->getType());
  auto Attr = mlir::stablehlo::DotDimensionNumbersAttr::get(
      theModule.getContext(), {}, {},
      {static_cast<int64_t>(LHSType->getConcreteShape().size() - 1)}, {0});
  auto ResultType = createRankedTensorTypeFromTensorType(
      NodeTensorType, *theModule.getContext());
  auto Op = builder.create<mlir::stablehlo::DotGeneralOp>(
      builder.getUnknownLoc(), ResultType,
      ValueRange{valueMap[node->getLHS()], valueMap[node->getRHS()]},
      ArrayRef<NamedAttribute>{
          builder.getNamedAttr("dot_dimension_numbers", Attr)});
  insertValueMapping(node, Op);
}

void StableHLOLoweringPass::visit(AddPtr node) {
  llvm::SmallVector<mlir::Value, 4> inputs = {valueMap[node->getLHS()],
                                              valueMap[node->getRHS()]};
  llvm::SmallVector<mlir::NamedAttribute, 4> attributes;
  auto op = builder.create<mlir::stablehlo::AddOp>(builder.getUnknownLoc(),
                                                   inputs, attributes);
  insertValueMapping(node, op);
}

void StableHLOLoweringPass::visit(PowPtr Node) {
  auto LHS = valueMap[Node->getOperand(0)];
  auto RHS = valueMap[Node->getOperand(1)];
  auto Op = builder.create<mlir::stablehlo::PowOp>(builder.getUnknownLoc(),
                                                   LHS.getType(), LHS, RHS);
  insertValueMapping(Node, Op);
}

void StableHLOLoweringPass::visit(ConvolutionPtr node) {
  mlir::Value input = valueMap[node->getInputValue()];
  mlir::Value weight = valueMap[node->getWeightValue()];
  auto convArgs = node->getArgs();

  // for dimensions NCHW padding_args's shape would  only be (2,2)
  auto padding_args = convArgs[3];
  auto paddingTensorType =
      mlir::RankedTensorType::get({2, 2}, builder.getIntegerType(64));
  mlir::DenseIntElementsAttr padding =
      mlir::DenseIntElementsAttr::get(paddingTensorType, padding_args);
  auto ResultType = createRankedTensorTypeFromTensorType(
      node->getType(), *theModule.getContext());
  llvm::SmallVector<mlir::Type, 1> resultTypes = {ResultType};
  auto windowStrides =
      convertI64VectorToDenseI64ArrayAttr(builder, convArgs[0]);
  auto lhsDilation =
      convertI64VectorToDenseI64ArrayAttr(builder, convArgs[1]); // lhs_dilation
  auto rhsDilation =
      convertI64VectorToDenseI64ArrayAttr(builder, convArgs[2]); // rhs_dilation
  // default 0 0
  auto window_reversal =
      builder.getDenseBoolArrayAttr({0, 0}); // window_reversal

  int64_t inputBatchDimension = 0;
  int64_t inputFeatureDimension = 1;
  llvm::SmallVector<int64_t, 2> inputSpatialDimensions = {2, 3};

  int64_t kernelInputFeatureDimension = 1;
  int64_t kernelOutputFeatureDimension = 0;
  llvm::SmallVector<int64_t, 2> kernelSpatialDimensions = {2, 3};

  int64_t outputBatchDimension = 0;
  int64_t outputFeatureDimension = 1;
  llvm::SmallVector<int64_t, 2> outputSpatialDimensions = {2, 3};

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

void StableHLOLoweringPass::visit(SqrtPtr node) {
  mlir::Value value = valueMap[node->getValue()];
  auto op =
      builder.create<mlir::stablehlo::SqrtOp>(builder.getUnknownLoc(), value);
  insertValueMapping(node, op);
}

void StableHLOLoweringPass::visit(ReversePtr Node) {
  mlir::Value Value = valueMap[Node->getOperand(0)];
  auto Axes = Node->getAxes();
  std::vector<int64_t> I64Axes = {Axes.begin(), Axes.end()};
  auto Op = builder.create<mlir::stablehlo::ReverseOp>(
      builder.getUnknownLoc(), Value,
      convertI64VectorToDenseI64ArrayAttr(builder, I64Axes));
  insertValueMapping(Node, Op);
}

void StableHLOLoweringPass::visit(SlicePtr node) {
  mlir::Value Value = valueMap[node->getOperand(0)];
  auto Starts = node->getStarts();
  std::vector<int64_t> I64Starts = {Starts.begin(), Starts.end()};
  auto Ends = node->getEnds();
  std::vector<int64_t> I64Ends = {Ends.begin(), Ends.end()};
  auto Strides = node->getStrides();
  std::vector<int64_t> I64Strides = {Strides.begin(), Strides.end()};
  auto Op = builder.create<mlir::stablehlo::SliceOp>(
      builder.getUnknownLoc(), Value,
      convertI64VectorToDenseI64ArrayAttr(builder, I64Starts),
      convertI64VectorToDenseI64ArrayAttr(builder, I64Ends),
      convertI64VectorToDenseI64ArrayAttr(builder, I64Strides));
  insertValueMapping(node, Op);
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
  auto outShape =
      SAFE_TYPE_DOWNCAST(node->getType(), TensorType)->getConcreteShape();
  std::vector<int64_t> outInt64Shape(outShape.size());
  std::transform(outShape.begin(), outShape.end(), outInt64Shape.begin(),
                 [](int val) { return static_cast<int64_t>(val); });

  mlir::RankedTensorType tensorType = mlir::RankedTensorType::get(
      mlir::ArrayRef<int64_t>(outInt64Shape), ElementType);
  mlir::DenseElementsAttr attr;
  if (ElementType.isa<mlir::FloatType>()) {
    float dimSizeValue = 1;
    for (auto dim_x : node->getDim()) {
      dimSizeValue *= inShape[dim_x];
    }
    // Create a DenseElementsAttr with the dimSize value
    attr = mlir::DenseElementsAttr::get(tensorType,
                                        llvm::ArrayRef<float>{dimSizeValue});
  } else if (ElementType.isa<mlir::IntegerType>()) {
    int64_t dimSizeValue = 1;
    for (auto dim_x : node->getDim()) {
      dimSizeValue *= inShape[dim_x];
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
  // Broadcast in dim to match ranks
  auto meanValue = builder.create<mlir::stablehlo::DivOp>(
      builder.getUnknownLoc(), reduceOp.getResult(0), dimSizeValue);
  insertValueMapping(node, meanValue);
}

void StableHLOLoweringPass::visit(SumPtr node) {
  auto CurrentBlock = builder.getInsertionBlock();
  bool keepdims = node->getKeepdims();
  mlir::Value value = valueMap[node->getValue()];
  auto dims = node->getDim();
  mlir::Type ElementType =
      value.getType().cast<mlir::TensorType>().getElementType();
  std::vector<int> inShape = node->getShape();
  mlir::RankedTensorType ResultType;
  if (keepdims == false)
    ResultType = createRankedTensorTypeFromTensorType(node->getType(),
                                                      *theModule.getContext());
  else {
    std::vector<int64_t> resultTensorShape;
    for (size_t Idx = 0; Idx < inShape.size(); ++Idx) {
      if (std::find(dims.begin(), dims.end(), Idx) == dims.end()) {
        resultTensorShape.push_back(static_cast<int64_t>(inShape[Idx]));
      }
    }

    ResultType = mlir::RankedTensorType::get(
        mlir::ArrayRef<int64_t>(resultTensorShape), ElementType);
  }
  mlir::Value initValue = builder.create<mlir::stablehlo::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getZeroAttr(ResultType.getElementType()));
  llvm::SmallVector<mlir::Value, 1> inputs = {value};
  llvm::SmallVector<mlir::Value, 1> initValues = {initValue};
  auto dimensions = builder.getDenseI64ArrayAttr(node->getDim());
  auto reduceOp = builder.create<mlir::stablehlo::ReduceOp>(
      builder.getUnknownLoc(), ResultType, inputs, initValues, dimensions);
  auto *Body = builder.createBlock(&reduceOp.getBodyRegion());
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
  auto res = reduceOp->getResult(0);
  std::vector<int64_t> reshapeTensorShape;
  if (keepdims) {
    for (size_t Idx = 0; Idx < inShape.size(); ++Idx) {
      if (std::find(dims.begin(), dims.end(), Idx) == dims.end()) {
        reshapeTensorShape.push_back(static_cast<int64_t>(inShape[Idx]));
      } else {
        reshapeTensorShape.push_back(1);
      }
    }

    mlir::RankedTensorType reshapeTensorType = mlir::RankedTensorType::get(
        mlir::ArrayRef<int64_t>(reshapeTensorShape), ElementType);
    auto op = builder.create<mlir::stablehlo::ReshapeOp>(
        builder.getUnknownLoc(), reshapeTensorType, reduceOp->getResult(0));
    insertValueMapping(node, op);
  } else {
    insertValueMapping(node, res);
  }
}

void StableHLOLoweringPass::visit(MaxPtr node) {
  auto CurrentBlock = builder.getInsertionBlock();
  bool keepdims = node->getKeepdims();
  mlir::Value value = valueMap[node->getValue()];
  auto dims = node->getDim();
  mlir::Type ElementType =
      value.getType().cast<mlir::TensorType>().getElementType();
  std::vector<int> inShape = node->getShape();
  mlir::RankedTensorType ResultType;
  if (keepdims == false)
    ResultType = createRankedTensorTypeFromTensorType(node->getType(),
                                                      *theModule.getContext());
  else {
    std::vector<int64_t> resultTensorShape;
    for (size_t Idx = 0; Idx < inShape.size(); ++Idx) {
      if (std::find(dims.begin(), dims.end(), Idx) == dims.end()) {
        resultTensorShape.push_back(static_cast<int64_t>(inShape[Idx]));
      }
    }

    ResultType = mlir::RankedTensorType::get(
        mlir::ArrayRef<int64_t>(resultTensorShape), ElementType);
  }
  mlir::Value initValue = builder.create<mlir::stablehlo::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getZeroAttr(ResultType.getElementType()));
  llvm::SmallVector<mlir::Value, 1> inputs = {value};
  llvm::SmallVector<mlir::Value, 1> initValues = {initValue};
  auto dimensions = builder.getDenseI64ArrayAttr(node->getDim());
  auto reduceOp = builder.create<mlir::stablehlo::ReduceOp>(
      builder.getUnknownLoc(), ResultType, inputs, initValues, dimensions);
  auto *Body = builder.createBlock(&reduceOp.getBodyRegion());
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
  auto res = reduceOp->getResult(0);
  std::vector<int64_t> reshapeTensorShape;
  if (keepdims) {
    for (size_t Idx = 0; Idx < inShape.size(); ++Idx) {
      if (std::find(dims.begin(), dims.end(), Idx) == dims.end()) {
        reshapeTensorShape.push_back(static_cast<int64_t>(inShape[Idx]));
      } else {
        reshapeTensorShape.push_back(1);
      }
    }

    mlir::RankedTensorType reshapeTensorType = mlir::RankedTensorType::get(
        mlir::ArrayRef<int64_t>(reshapeTensorShape), ElementType);
    auto op = builder.create<mlir::stablehlo::ReshapeOp>(
        builder.getUnknownLoc(), reshapeTensorType, reduceOp->getResult(0));
    insertValueMapping(node, op);
  } else {
    insertValueMapping(node, res);
  }
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
  auto outShape =
      SAFE_TYPE_DOWNCAST(node->getType(), TensorType)->getConcreteShape();
  std::vector<int64_t> outInt64Shape(outShape.size());
  std::transform(outShape.begin(), outShape.end(), outInt64Shape.begin(),
                 [](int val) { return static_cast<int64_t>(val); });

  mlir::RankedTensorType tensorType = mlir::RankedTensorType::get(
      mlir::ArrayRef<int64_t>(outInt64Shape), ElementType);
  mlir::DenseElementsAttr attr;
  mlir::DenseElementsAttr correction_attr;

  int ddof = node->getDdof();
  if (ElementType.isa<mlir::FloatType>()) {
    float dimSizeValue = 1;
    float correction;

    for (auto dim_x : node->getDim()) {
      dimSizeValue *= inShape[dim_x];
    }
    correction = dimSizeValue - ddof;
    // Create a DenseElementsAttr with the dimSize value
    attr = mlir::DenseElementsAttr::get(tensorType,
                                        llvm::ArrayRef<float>{dimSizeValue});
    correction_attr = mlir::DenseElementsAttr::get(
        tensorType, llvm::ArrayRef<float>{correction});
  } else if (ElementType.isa<mlir::IntegerType>()) {
    int64_t dimSizeValue = 1;
    int64_t correction;

    for (auto dim_x : node->getDim()) {
      dimSizeValue *= inShape[dim_x];
    }
    correction = dimSizeValue - ddof;
    // Create a DenseElementsAttr with the dimSize value
    attr = mlir::DenseElementsAttr::get(tensorType,
                                        llvm::ArrayRef<int64_t>{dimSizeValue});
    correction_attr = mlir::DenseElementsAttr::get(
        tensorType, llvm::ArrayRef<int64_t>{correction});
  } else {
    throw std::runtime_error("Unsupported data type lowering VarianceOp.");
  }

  // Create a ConstantOp with the tensor type and attribute
  auto dimSizeValue = builder.create<mlir::stablehlo::ConstantOp>(
      builder.getUnknownLoc(), tensorType, attr);
  // Broadcast in dim to match ranks
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
  mlir::RankedTensorType ScalarFloatType =
      mlir::RankedTensorType::get(mlir::ArrayRef<int64_t>({}), ElementType);

  std::vector<int64_t> broad_dims;
  auto feature_dims = node->getDim();
  for (size_t i = 0; i < inShape.size(); ++i) {
    if (std::find(feature_dims.begin(), feature_dims.end(), i) ==
        feature_dims.end()) {
      broad_dims.push_back(i);
    }
  }

  mlir::DenseI64ArrayAttr axes_attr = builder.getDenseI64ArrayAttr(broad_dims);
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
  mlir::FloatAttr epsilon = builder.getF32FloatAttr(0.00001);
  mlir::IntegerAttr feature_index = builder.getI64IntegerAttr(1);

  auto op = builder.create<mlir::stablehlo::BatchNormInferenceOp>(
      builder.getUnknownLoc(), resultTypes, value, scale, offset, mean,
      variance, epsilon, feature_index);
  insertValueMapping(node, op);
}

void StableHLOLoweringPass::visit(Maxpool2dPtr node) {
  auto CurrentBlock = builder.getInsertionBlock();
  mlir::Value value = valueMap[node->getValue()];
  auto MaxpoolArgs = node->getArgs();
  auto window_dimensions =
      convertI64VectorToDenseI64ArrayAttr(builder, MaxpoolArgs[0]);
  auto window_strides =
      convertI64VectorToDenseI64ArrayAttr(builder, MaxpoolArgs[1]);

  auto base_dilations =
      convertI64VectorToDenseI64ArrayAttr(builder, MaxpoolArgs[2]);
  auto window_dilations =
      convertI64VectorToDenseI64ArrayAttr(builder, MaxpoolArgs[3]);
  auto padding_args = MaxpoolArgs[4];
  auto paddingTensorType =
      mlir::RankedTensorType::get({4, 2}, builder.getIntegerType(64));
  mlir::DenseIntElementsAttr padding =
      mlir::DenseIntElementsAttr::get(paddingTensorType, padding_args);
  auto ResultType = createRankedTensorTypeFromTensorType(
      node->getType(), *theModule.getContext());
  mlir::Value initValue = builder.create<mlir::stablehlo::ConstantOp>(
      builder.getUnknownLoc(), builder.getZeroAttr(builder.getF32Type()));
  auto ReduceWindowOp = builder.create<mlir::stablehlo::ReduceWindowOp>(
      builder.getUnknownLoc(), ResultType, value, initValue, window_dimensions,
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

void StableHLOLoweringPass::visit(ScatterAddMaxPtr node) {
  mlir::Value Operand = valueMap[node->getOperand(0)];
  mlir::Value Source = valueMap[node->getOperand(1)];
  mlir::Value InitValue = valueMap[node->getOperand(2)];
  // auto Op = builder.create<mlir::stablehlo::SelectAndScatterOp>(
  //     builder.getUnknownLoc(), value, indices, updates);
  // insertValueMapping(node, Op);
  auto *CurrentBlock = builder.getInsertionBlock();
  mlir::stablehlo::SelectAndScatterOp();
  auto ScatterAddMaxArgs = node->getArgs();
  auto WindowDimensions =
      convertI64VectorToDenseI64ArrayAttr(builder, ScatterAddMaxArgs[0]);
  auto WindowStrides =
      convertI64VectorToDenseI64ArrayAttr(builder, ScatterAddMaxArgs[1]);
  auto PaddingArgs = ScatterAddMaxArgs[2];
  auto PaddingTensorType =
      mlir::RankedTensorType::get({4, 2}, builder.getIntegerType(64));
  mlir::DenseIntElementsAttr Padding =
      mlir::DenseIntElementsAttr::get(PaddingTensorType, PaddingArgs);
  auto Op = builder.create<mlir::stablehlo::SelectAndScatterOp>(
      builder.getUnknownLoc(), Operand.getType(), Operand, Source, InitValue,
      WindowDimensions, WindowStrides, Padding);

  auto *Body = builder.createBlock(&Op.getSelect());
  builder.setInsertionPointToEnd(Body);
  auto ElementType =
      Operand.getType().cast<mlir::TensorType>().getElementType();
  auto CompareArg0 = Body->addArgument(
      mlir::RankedTensorType::get({}, ElementType), builder.getUnknownLoc());
  auto CompareArg1 = Body->addArgument(
      mlir::RankedTensorType::get({}, ElementType), builder.getUnknownLoc());
  auto CompareDirection = mlir::stablehlo::ComparisonDirection::GE;
  mlir::stablehlo::ComparisonType CompareType;
  if (auto OperandTensorType =
          asType<TensorType>(node->getOperand(0)->getType())) {
    auto ElementType = OperandTensorType->getElementType();
    if (ElementType->isFloatType())
      CompareType = mlir::stablehlo::ComparisonType::FLOAT;
  }
  CompareType = mlir::stablehlo::ComparisonType::SIGNED;
  auto CmpOp = builder.create<mlir::stablehlo::CompareOp>(
      builder.getUnknownLoc(), CompareArg0, CompareArg1, CompareDirection,
      CompareType);
  builder.create<mlir::stablehlo::ReturnOp>(builder.getUnknownLoc(),
                                            CmpOp.getResult());
  builder.setInsertionPointToEnd(CurrentBlock);

  auto *ScatterBody = builder.createBlock(&Op.getScatter());
  builder.setInsertionPointToEnd(ScatterBody);
  auto ScatterArg0 = ScatterBody->addArgument(
      mlir::RankedTensorType::get({}, ElementType), builder.getUnknownLoc());
  auto ScatterArg1 = ScatterBody->addArgument(
      mlir::RankedTensorType::get({}, ElementType), builder.getUnknownLoc());
  auto AddOp = builder.create<mlir::stablehlo::AddOp>(builder.getUnknownLoc(),
                                                      ScatterArg0, ScatterArg1);
  builder.create<mlir::stablehlo::ReturnOp>(builder.getUnknownLoc(),
                                            AddOp.getResult());
  builder.setInsertionPointToEnd(CurrentBlock);

  insertValueMapping(node, Op);
}

void StableHLOLoweringPass::visit(Avgpool2dPtr node) {
  auto CurrentBlock = builder.getInsertionBlock();
  mlir::Value value = valueMap[node->getValue()];
  auto AvgpoolArgs = node->getArgs();
  auto window_dimensions =
      convertI64VectorToDenseI64ArrayAttr(builder, AvgpoolArgs[0]);
  auto window_strides =
      convertI64VectorToDenseI64ArrayAttr(builder, AvgpoolArgs[1]);

  auto base_dilations =
      convertI64VectorToDenseI64ArrayAttr(builder, AvgpoolArgs[2]);
  auto window_dilations =
      convertI64VectorToDenseI64ArrayAttr(builder, AvgpoolArgs[3]);
  auto padding_args = AvgpoolArgs[4];
  auto paddingTensorType =
      mlir::RankedTensorType::get({4, 2}, builder.getIntegerType(64));
  mlir::DenseIntElementsAttr padding =
      mlir::DenseIntElementsAttr::get(paddingTensorType, padding_args);
  auto ResultType = createRankedTensorTypeFromTensorType(
      node->getType(), *theModule.getContext());
  mlir::Value initValue = builder.create<mlir::stablehlo::ConstantOp>(
      builder.getUnknownLoc(), builder.getZeroAttr(builder.getF32Type()));
  auto ReduceWindowOp = builder.create<mlir::stablehlo::ReduceWindowOp>(
      builder.getUnknownLoc(), ResultType, value, initValue, window_dimensions,
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
      builder.create<mlir::stablehlo::AddOp>(builder.getUnknownLoc(), AddArgs);
  builder.create<mlir::stablehlo::ReturnOp>(builder.getUnknownLoc(),
                                            AddResult.getResult());
  builder.setInsertionPointToEnd(CurrentBlock);

  auto kernel_dim = AvgpoolArgs[0];
  auto kernel_accum = static_cast<float>(std::accumulate(
      kernel_dim.begin(), kernel_dim.end(), 1, std::multiplies<int64_t>()));
  auto outShape =
      SAFE_TYPE_DOWNCAST(node->getType(), TensorType)->getConcreteShape();
  std::vector<int64_t> outInt64Shape(outShape.size());
  std::transform(outShape.begin(), outShape.end(), outInt64Shape.begin(),
                 [](int val) { return static_cast<int64_t>(val); });

  mlir::RankedTensorType tensorType = mlir::RankedTensorType::get(
      mlir::ArrayRef<int64_t>(outInt64Shape), ElementType);
  mlir::DenseElementsAttr attr;
  attr = mlir::DenseElementsAttr::get(tensorType,
                                      llvm::ArrayRef<float>{kernel_accum});
  auto kernel_size_size_value = builder.create<mlir::stablehlo::ConstantOp>(
      builder.getUnknownLoc(), tensorType, attr);
  auto res = builder.create<mlir::stablehlo::DivOp>(
      builder.getUnknownLoc(), ReduceWindowOp->getResult(0),
      kernel_size_size_value);
  insertValueMapping(node, res);
}

void StableHLOLoweringPass::visit(BroadcastPtr Node) {
  mlir::Value Value = valueMap[Node->getOperand(0)];
  auto Shape = Node->getBroadCastShape();
  auto InputTensorType = asType<TensorType>(Node->getOperand(0)->getType());

  auto PrevShape = InputTensorType->getConcreteShape();
  std::vector<int64_t> BroadCastDimensions;
  size_t OperandIndex = 0;
  size_t OperandRank = PrevShape.size();
  size_t TargetRank = Shape.size();
  for (size_t Idx = 0; Idx < Shape.size(); ++Idx) {
    if (OperandIndex < OperandRank && (PrevShape[OperandIndex] == 1 ||
                                       Shape[Idx] == PrevShape[OperandIndex])) {
      BroadCastDimensions.push_back(Idx);
      ++OperandIndex;
    }
  }
  if (OperandIndex < OperandRank) {
    throw std::runtime_error("Invalid broadcast shape");
  }
  std::vector<ValuePtr> ResultValueShape;
  for (auto Dim : Shape) {
    ResultValueShape.push_back(Literal::create(static_cast<int>(Dim)));
  }
  auto ResultTensorType =
      TensorType::create(InputTensorType->getElementType(), ResultValueShape);
  if (Node->isDynamicBroadcast()) {
    auto NodeTensorType = asType<TensorType>(Node->getType());
    auto NodeShape = NodeTensorType->getConcreteShape();
    std::vector<int> I64NodeShape;
    for (auto Dim : NodeShape) {
      I64NodeShape.push_back(Dim);
    }
    auto OutputDimensionType = TensorType::create(
        IntTypePtr::get(),
        {Literal::create(static_cast<int>(I64NodeShape.size()))});
    auto OutputDimensionAttr = mlir::DenseElementsAttr::get(
        createRankedTensorTypeFromTensorType(OutputDimensionType,
                                             *theModule->getContext()),
        ArrayRef<int>(I64NodeShape));
    auto OutputDimension = builder.create<mlir::stablehlo::ConstantOp>(
        builder.getUnknownLoc(),
        createRankedTensorTypeFromTensorType(OutputDimensionType,
                                             *theModule->getContext()),
        OutputDimensionAttr);
    auto Op = builder.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
        builder.getUnknownLoc(),
        createRankedTensorTypeFromTensorType(NodeTensorType,
                                             *theModule->getContext()),
        Value, OutputDimension,
        builder.getDenseI64ArrayAttr(BroadCastDimensions));
    insertValueMapping(Node, Op);
  } else {
    auto Op = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        builder.getUnknownLoc(),
        createRankedTensorTypeFromTensorType(ResultTensorType,
                                             *theModule->getContext()),
        Value, ArrayRef<int64_t>(BroadCastDimensions));
    insertValueMapping(Node, Op);
  }
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

void StableHLOLoweringPass::visit(SelectPtr Node) {
  auto Condition = valueMap[Node->getOperand(0)];
  auto TrueValue = valueMap[Node->getOperand(1)];
  auto FalseValue = valueMap[Node->getOperand(2)];
  auto Op = builder.create<mlir::stablehlo::SelectOp>(
      builder.getUnknownLoc(), TrueValue.getType(), Condition, TrueValue,
      FalseValue);
  insertValueMapping(Node, Op);
}

void StableHLOLoweringPass::visit(ConcatPtr Node) {
  llvm::SmallVector<mlir::Value, 4> inputs;
  for (auto value : Node->getInputs()) {
    inputs.push_back(valueMap[value]);
  }
  auto op = builder.create<mlir::stablehlo::ConcatenateOp>(
      builder.getUnknownLoc(), inputs, Node->getDim());
  insertValueMapping(Node, op);
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

void StableHLOLoweringPass::visit(ExpPtr node) {
  mlir::Value value = valueMap[node->getOperand(0)];
  auto op =
      builder.create<mlir::stablehlo::ExpOp>(builder.getUnknownLoc(), value);
  insertValueMapping(node, op);
}

void StableHLOLoweringPass::visit(TanhPtr node) {
  mlir::Value value = valueMap[node->getOperand(0)];
  auto op =
      builder.create<mlir::stablehlo::TanhOp>(builder.getUnknownLoc(), value);
  insertValueMapping(node, op);
}

void StableHLOLoweringPass::visit(NegPtr node) {
  mlir::Value value = valueMap[node->getOperand(0)];
  auto op =
      builder.create<mlir::stablehlo::NegOp>(builder.getUnknownLoc(), value);
  insertValueMapping(node, op);
}

void StableHLOLoweringPass::visit(DivPtr node) {
  mlir::Value lhs = valueMap[node->getOperand(0)];
  mlir::Value rhs = valueMap[node->getOperand(1)];
  auto op =
      builder.create<mlir::stablehlo::DivOp>(builder.getUnknownLoc(), lhs, rhs);
  insertValueMapping(node, op);
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
  auto name = module->getName();
  if (name == "forward" || name == "backward") 
  	visualizeModule(loweringPass->module(), module->getName(), context);
  return mlirModuleToString(loweringPass->module());
}

std::string mlirModuleToString(mlir::ModuleOp module) {
  std::string str;
  llvm::raw_string_ostream os(str);
  module.print(os);
  return os.str();
}

} // namespace ainl::ir
