#include "ailang/Transforms/Autodiff.h"
#include "ailang/AST/ASTNode.h"
#include "ailang/IR/Container.h"
#include "ailang/IR/Function.h"
#include "ailang/IR/IRVisitor.h"
#include "ailang/IR/Literal.h"
#include "ailang/IR/Node.h"
#include "ailang/IR/Type.h"
#include "ailang/IR/Value.h"
#include "ailang/Transforms/Pass.h"
#include "ailang/Transforms/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace ainl::ir;

void ainl::ir::autodiffOnModule(ModulePtr M) {
  auto AD = std::make_unique<AutoDiff>(M);
  M->setName("backward");
  LOG_DEBUG("%s", "Running autodiff on module");
  AD->run(M);
  LOG_DEBUG("%s", "Autodiff finished");
}

void AutoDiff::run(ModulePtr M) {
  auto Graph = M->getGraph();
  M->setName("backward");
  std::vector<NodePtr> Nodes;
  for (auto *Block : *Graph) {
    for (auto *Node : *Block) {
      Nodes.push_back(Node);
    }
  }
  std::reverse(Nodes.begin(), Nodes.end());
  for (auto *Node : Nodes) {
    LOG_DEBUG("differentiating: %s", std::string(*Node).c_str());
    Node->accept(this);
  }
}

void AutoDiff::visit(ParamPtr Node) {
  auto Params = Node->getParams();
  auto NumParams = Params.size();
  std::vector<ValuePtr> ModuleReturns;
  if (asValueType<TupleContainer>(ReturnValue)) {
    throw std::runtime_error(
        "Return value must be a single value when differentiating a function.");
  } else {
    auto ReturnValueType = asType<TensorType>(ReturnValue->getType());
    if (!ReturnValueType->getConcreteShape().empty()) {
      throw std::runtime_error(
          "Return value must be a scalar when differentiating a function.");
    }
    ModuleReturns.push_back(ReturnValue);
  }
  if (Params.size() > 1) {
    for (auto *Param : Params) {
      if (hasAdjoint(Param)) {
        auto *Adjoint = getAdjoint(Param);
        ModuleReturns.push_back(Adjoint);
      } else {
        auto ParamType = asType<TensorType>(Param->getType());
        auto ParamShape = ParamType->getConcreteShape();
        if (ParamShape.empty()) {
          auto *Adjoint = Module->create<ConstantDef>(
              ParamType, TupleContainer::create({Literal::create(0.f)}));
          setAdjoint(Param, Adjoint);
          ModuleReturns.push_back(Adjoint);
        } else {
          std::vector<ValuePtr> InitGradients;
          size_t NumElements = 1;
          for (auto Axis : ParamShape) {
            NumElements *= Axis;
          }
          for (size_t Idx = 0; Idx < NumElements; Idx++) {
            InitGradients.push_back(Literal::create(0.f));
          }
          auto *ParamGradient = Module->create<ConstantDef>(
              ParamType, TupleContainer::create(InitGradients));
          setAdjoint(Param, ParamGradient);
          ModuleReturns.push_back(ParamGradient);
        }
      }
    }
    auto *Tuple = TupleContainer::create(ModuleReturns);
    Module->setReturnType(Tuple->getType());
    Module->create<ReturnOp>(Tuple);
  } else {
    assert(Params.size() == 1 && "Params size must at least be 1");
    if (hasAdjoint(Params[0])) {
      auto *GradientReturn = getAdjoint(Params[0]);
      ModuleReturns.push_back(GradientReturn);
      auto *Tuple = TupleContainer::create(ModuleReturns);
      Module->setReturnType(Tuple->getType());
      Module->create<ReturnOp>(Tuple);
    } else {
      auto ParamType = asType<TensorType>(Params[0]->getType());
      auto ParamShape = ParamType->getConcreteShape();
      if (ParamShape.empty()) {
        auto *Adjoint = Module->create<ConstantDef>(
            ParamType, TupleContainer::create({Literal::create(0.f)}));
        setAdjoint(Params[0], Adjoint);
        ModuleReturns.push_back(Adjoint);
      } else {
        std::vector<ValuePtr> InitGradients;
        size_t NumElements = 1;
        for (auto Axis : ParamShape) {
          NumElements *= Axis;
        }
        for (size_t Idx = 0; Idx < NumElements; Idx++) {
          InitGradients.push_back(Literal::create(0.f));
        }
        auto *ParamGradient = Module->create<ConstantDef>(
            ParamType, TupleContainer::create(InitGradients));
        setAdjoint(Params[0], ParamGradient);
        ModuleReturns.push_back(ParamGradient);
      }
    }
  }
}

void AutoDiff::visit(ReturnOpPtr Node) {
  auto *Value = Node->getReturnValue();
  ReturnValue = Value;
  Module->remove(Node);
  if (asValueType<TupleContainer>(Value)) {
    auto *Tuple = asValueType<TupleContainer>(Value);
    for (auto *Item : Tuple->getValues()) {
      auto ItemType = asType<TensorType>(Item->getType());
      auto ItemShape = ItemType->getConcreteShape();
      std::vector<ValuePtr> InitGradients;
      if (ItemShape.empty()) {
        InitGradients.push_back(Literal::create(1.f));
      } else {
        size_t NumElements = 1;
        for (auto Axis : ItemShape) {
          NumElements *= Axis;
        }
        for (size_t Idx = 0; Idx < NumElements; Idx++) {
          InitGradients.push_back(Literal::create(0.f));
        }
      }
      auto *ItemGradient = Module->create<ConstantDef>(
          ItemType, TupleContainer::create(InitGradients));
      setAdjoint(Item, ItemGradient);
    }
  } else {
    auto ItemType = asType<TensorType>(Value->getType());
    auto ItemShape = ItemType->getConcreteShape();
    std::vector<ValuePtr> InitGradients;
    if (ItemShape.empty()) {
      InitGradients.push_back(Literal::create(1.f));
    } else {
      size_t NumElements = 1;
      for (auto Axis : ItemShape) {
        NumElements *= Axis;
      }
      for (size_t Idx = 0; Idx < NumElements; Idx++) {
        InitGradients.push_back(Literal::create(0.f));
      }
    }
    auto *ItemGradient = Module->create<ConstantDef>(
        ItemType, TupleContainer::create(InitGradients));
    setAdjoint(Value, ItemGradient);
  }
}

void AutoDiff::visit(SqrtPtr Node) {
  auto *Value = Node->getOperand(0);
  auto *Adjoint = getAdjoint(Node);
  auto *AdjointNode = Module->create<Div>(
      Node->getType(), Adjoint,
      Module->create<Mul>(Node->getType(),
                          Module->createConstantValue(2.f, Node->getType()),
                          Node));
  setAdjoint(Value, AdjointNode);
}

void AutoDiff::visit(SumPtr Node) {
  auto *Value = Node->getOperand(0);
  auto Dim = Node->getDim();
  auto *Adjoint = getAdjoint(Node);
  auto AdjointType = asType<TensorType>(Adjoint->getType());
  auto ValueType = asType<TensorType>(Value->getType());
  auto *BroadcastedAdjointValue = Module->create<Broadcast>(
      ValueType, Adjoint, ValueType->getConcreteShape());
  setAdjoint(Value, BroadcastedAdjointValue);
}

void AutoDiff::visit(MaxPtr Node) {
  auto *Value = Node->getOperand(0);
  auto Dim = Node->getDim();
  auto ValueTensorType = asType<TensorType>(Value->getType());
  auto *BroadcastedMaxValue = Module->create<Broadcast>(
      ValueTensorType, Node, ValueTensorType->getConcreteShape());
  auto *CompareValue = Module->create<CompareOp>(
      Value->getType(), Value, BroadcastedMaxValue, CompareOp::CompareType::GE);
  auto *Adjoint = getAdjoint(Node);
  auto *BroadcastedAdjoint = Module->create<Broadcast>(
      ValueTensorType, Adjoint, ValueTensorType->getConcreteShape());
  auto *SelectedAdjoint = Module->create<Select>(
      BroadcastedAdjoint->getType(), CompareValue, BroadcastedAdjoint,
      Module->createConstantValue(0.f, BroadcastedAdjoint->getType()));
  setAdjoint(Value, SelectedAdjoint);
}

void AutoDiff::visit(ExpPtr Node) {
  auto *Value = Node->getOperand(0);
  auto *Adjoint = getAdjoint(Node);
  auto *AdjointNode = Module->create<Mul>(Node->getType(), Node, Adjoint);
  setAdjoint(Value, AdjointNode);
}

void AutoDiff::visit(TanhPtr Node) {
  auto *Value = Node->getOperand(0);
  auto *Adjoint = getAdjoint(Node);
  auto *OneConstant = Module->createConstantValue(1.f, Node->getType());
  auto *NegTanh2 = Module->create<Neg>(
      Node->getType(), Module->create<Mul>(Node->getType(), Node, Node));
  auto *AdjointNode = Module->create<Mul>(
      Node->getType(), Adjoint,
      Module->create<Add>(Node->getType(), OneConstant, NegTanh2));
  setAdjoint(Value, AdjointNode);
}

void AutoDiff::visit(ReluPtr Node) {
  auto *Value = Node->getOperand(0);
  auto *ComapreResult = Module->create<CompareOp>(
      Node->getType(), Value, Module->createConstantValue(0.f, Node->getType()),
      CompareOp::CompareType::GT);
  auto *Adjoint = getAdjoint(Node);
  auto *SelectResult =
      Module->create<Select>(Node->getType(), ComapreResult, Adjoint,
                             Module->createConstantValue(0.f, Node->getType()));
  setAdjoint(Value, SelectResult);
}

void AutoDiff::visit(MeanPtr Node) {
  auto *Value = Node->getOperand(0);
  auto Dim = Node->getDim();
  auto *Adjoint = getAdjoint(Node);
  auto AdjointType = asType<TensorType>(Adjoint->getType());
  auto ValueType = asType<TensorType>(Value->getType());
  auto *BroadcastedAdjointValue = Module->create<Broadcast>(
      ValueType, Adjoint, ValueType->getConcreteShape());
  float NumElements = 1.f;
  for (auto Axis : Dim) {
    NumElements *= ValueType->getConcreteShape()[Axis];
  }
  auto *Divisor = Module->create<Broadcast>(
      ValueType, Module->createConstantValue(NumElements, Node->getType()),
      ValueType->getConcreteShape());
  auto *AdjointNode =
      Module->create<Div>(ValueType, BroadcastedAdjointValue, Divisor);
  setAdjoint(Value, AdjointNode);
}

void AutoDiff::visit(VariancePtr Node) {
  auto *Value = Node->getOperand(0);
  auto Dim = Node->getDim();
  auto *MeanValue = Module->create<Mean>(Node->getType(), Value, Dim);
  auto ValueType = asType<TensorType>(Value->getType());
  auto *BroadcastedMeanValue = Module->create<Broadcast>(
      ValueType, MeanValue, ValueType->getConcreteShape());
  auto *SubtractValue = Module->create<Add>(
      Value->getType(), Value,
      Module->create<Neg>(Value->getType(), BroadcastedMeanValue));
  auto *Adjoint = getAdjoint(Node);
  auto *BroadcastedAdjointValue = Module->create<Broadcast>(
      ValueType, Adjoint, ValueType->getConcreteShape());
  float NumElements = 1.f;
  for (auto Axis : Dim) {
    NumElements *= ValueType->getConcreteShape()[Axis];
  }
  int Ddof = Node->getDdof();
  if (Ddof) {
    assert(NumElements > 1.f && "NumElements must be greater than 1");
    auto *Divisor =
        Module->createConstantValue(NumElements - 1, Value->getType());
    auto *MulDivisor = Module->create<Div>(
        Value->getType(), Module->createConstantValue(2.f, Value->getType()),
        Divisor);
    auto *MulSubtractValue =
        Module->create<Mul>(Value->getType(), MulDivisor, SubtractValue);
    auto *AdjointNode = Module->create<Mul>(
        Value->getType(), BroadcastedAdjointValue, MulSubtractValue);
    setAdjoint(Value, AdjointNode);
  } else {
    auto *Divisor = Module->createConstantValue(NumElements, Value->getType());
    auto *MulDivisor = Module->create<Div>(
        Value->getType(), Module->createConstantValue(2.f, Value->getType()),
        Divisor);
    auto *MulSubtractValue =
        Module->create<Mul>(Value->getType(), MulDivisor, SubtractValue);
    auto *AdjointNode = Module->create<Mul>(
        Value->getType(), BroadcastedAdjointValue, MulSubtractValue);
    setAdjoint(Value, AdjointNode);
  }
}

void AutoDiff::visit(BatchNorm2dPtr Node) {
  auto *InputValue = Node->getValue();
  auto *Scale = Node->getScale();
  auto *Mean = Node->getMean();
  auto *Variance = Node->getVariance();
  auto *Offset = Node->getOffset();
  auto InputValueType = asType<TensorType>(InputValue->getType());
  auto *BroadcastedMean = Module->create<Broadcast>(
      Node->getType(), Mean, InputValueType->getConcreteShape());
  auto *BroadcastedVariance = Module->create<Broadcast>(
      Node->getType(), Variance, InputValueType->getConcreteShape());
  auto *BroadcastedScale = Module->create<Broadcast>(
      Node->getType(), Scale, InputValueType->getConcreteShape());
  auto *BroadcastedOffset = Module->create<Broadcast>(
      Node->getType(), Offset, InputValueType->getConcreteShape());
  auto *SqrtVariance = Module->create<Pow>(
      Node->getType(),
      Module->create<Add>(Node->getType(), BroadcastedVariance,
                          Module->createConstantValue(1e-5f, Node->getType())),
      Module->createConstantValue(0.5f, Node->getType()));
  auto *Adjoint = getAdjoint(Node);
  auto *ScalingDivSqrtVariance =
      Module->create<Div>(Node->getType(), BroadcastedScale, SqrtVariance);
  auto *InputValueAdjoint =
      Module->create<Mul>(Node->getType(), Adjoint, ScalingDivSqrtVariance);
  auto *MeanAdjoint = Module->create<Mul>(
      Node->getType(), Adjoint,
      Module->create<Neg>(Node->getType(), ScalingDivSqrtVariance));
  setAdjoint(InputValue, InputValueAdjoint);
  setAdjoint(Mean, MeanAdjoint);
  auto *InputValueSubMean = Module->create<Add>(
      Node->getType(), InputValue,
      Module->create<Neg>(Node->getType(), BroadcastedMean));
  auto *InputValuePow = Module->create<Pow>(
      Node->getType(),
      Module->create<Add>(Node->getType(), BroadcastedVariance,
                          Module->createConstantValue(1e-5f, Node->getType())),
      Module->createConstantValue(-1.5f, Node->getType()));
  auto *VarAdjoint = Module->create<Mul>(
      Node->getType(), Adjoint,
      Module->create<Mul>(
          Node->getType(), BroadcastedScale,
          Module->create<Mul>(Node->getType(), InputValueSubMean,
                              Module->create<Mul>(Node->getType(),
                                                  Module->createConstantValue(
                                                      -0.5f, Node->getType()),
                                                  InputValuePow))));
  setAdjoint(Variance, VarAdjoint);
}

void AutoDiff::visit(Maxpool2dPtr Node) {
  // layout: NxCxHxW: 0, 1, 2, 3
  // Batched Scatter and Select_And_Scatter is not supported
  // In Recent IREE Release, We handle each batch separately
  auto MaxpoolArgs = Node->getArgs();
  auto WindowDimensions = MaxpoolArgs[0];
  auto WindowStrides = MaxpoolArgs[1];
  auto PaddingArgs = MaxpoolArgs[4];
  auto *Value = Node->getOperand(0);
  auto ValueTensorType = asType<TensorType>(Value->getType());
  auto ValueShape = ValueTensorType->getConcreteShape();
  auto *Adjoint = getAdjoint(Node);
  auto *InitZeroValue = Module->create<ConstantDef>(
      TensorType::create(ValueTensorType->getElementType(), {}),
      TupleContainer::create({Literal::create(0.f)}));
  auto *ScatterOp = Module->create<ScatterAddMax>(
      ValueTensorType, Value, Adjoint, InitZeroValue, WindowDimensions,
      WindowStrides, PaddingArgs);
  setAdjoint(Value, ScatterOp);
}

void AutoDiff::visit(ConvolutionPtr Node) {
  auto *InputValue = Node->getOperand(0);
  auto InputTensorShape =
      asType<TensorType>(InputValue->getType())->getConcreteShape();
  auto InputSize = InputTensorShape[2];
  assert(InputSize == InputTensorShape[3] && "Conv input must be square");
  auto *Weight = Node->getOperand(1);
  auto WeightShape = asType<TensorType>(Weight->getType())->getConcreteShape();
  auto WeightValueShape = asType<TensorType>(Weight->getType())->getShape();
  auto KernelSize = WeightShape[2];
  assert(KernelSize == WeightShape[3] && "Conv kernel must be square");
  auto ConvArgs = Node->getArgs();
  auto WindowStrides = ConvArgs[0];
  auto KernelStride = WindowStrides[0];
  assert(KernelStride == WindowStrides[1] && "Conv stride must be square");
  auto LhsDilation = ConvArgs[1];
  auto RhsDilation = ConvArgs[2];
  auto PaddingArgs = ConvArgs[3];
  auto WindowReversal = ConvArgs[4];
  auto *Adjoint = getAdjoint(Node);
  auto NodeTensorShape =
      asType<TensorType>(Node->getType())->getConcreteShape();
  auto NodeSize = NodeTensorShape[2];
  assert(NodeSize == NodeTensorShape[3] && "Conv output must be square");
  // Reverse the weight
  auto *ReversedWeight = Module->create<Reverse>(Weight->getType(), Weight,
                                                 std::vector<int>{2, 3});
  auto TransposedReversedWeightType = TensorType::create(
      asType<TensorType>(ReversedWeight->getType())->getElementType(),
      {WeightValueShape[1], WeightValueShape[0], WeightValueShape[2],
       WeightValueShape[3]});
  auto *TransposedReversedWeight =
      Module->create<Transpose>(TransposedReversedWeightType, ReversedWeight,
                                std::vector<int>{1, 0, 2, 3});
  // Compute new conv parameters
  std::vector<int64_t> TransposedStrides;
  for (size_t Idx = 0; Idx < WindowStrides.size(); Idx++) {
    TransposedStrides.push_back(1);
  }
  std::vector<int64_t> TransposedLhsDilation;
  for (size_t Idx = 0; Idx < WindowStrides.size(); Idx++) {
    TransposedLhsDilation.push_back(KernelStride);
  }
  std::vector<int64_t> TransposedPaddingArgs;
  int64_t PaddingBefore;
  int64_t PaddingAfter;
  PaddingBefore = KernelSize - 1 - PaddingArgs[0];
  PaddingAfter = InputSize + KernelSize - 1 - PaddingBefore -
                 ((NodeSize - 1) * KernelStride + 1);
  // for (size_t Idx = 0; Idx < PaddingArgs.size(); Idx++) {
  //   TransposedPaddingArgs.push_back(KernelSize - 1 - PaddingArgs[Idx]);
  // }
  TransposedPaddingArgs.push_back(PaddingBefore);
  TransposedPaddingArgs.push_back(PaddingAfter);
  TransposedPaddingArgs.push_back(PaddingBefore);
  TransposedPaddingArgs.push_back(PaddingAfter);
  auto *ValueAdjoint = Module->create<Convolution>(
      InputValue->getType(), Adjoint, TransposedReversedWeight,
      TransposedStrides, TransposedLhsDilation, RhsDilation,
      TransposedPaddingArgs, WindowReversal);
  setAdjoint(InputValue, ValueAdjoint);
}

void AutoDiff::visit(ConstantDefPtr Node) {
  // auto *Value = Node->getOperand(0);
  // auto ResultTensorType = asType<TensorType>(Node->getType());
  // // derivative must be float type
  // std::vector<ValuePtr> FloatValues;
  // auto *FloatContainer = asValueType<TupleContainer>(Node->getOperand(0));
  // for (auto &Value : FloatContainer->getValues()) {
  //   FloatValues.push_back(Literal::create(0.f));
  // }
  // auto *TangentNode = Module->createAfter<ConstantDef>(
  //     Node, Node->getType(), TupleContainer::create(FloatValues));
  // setTangent(Node, TangentNode);
}

void AutoDiff::visit(AddPtr Node) {
  auto *Left = Node->getOperand(0);
  auto *Right = Node->getOperand(1);
  auto *Adjoint = getAdjoint(Node);
  auto *LeftAdjoint =
      Module->create<Add>(Node->getType(), Adjoint,
                          Module->createConstantValue(0.f, Node->getType()));
  auto *RightAdjoint =
      Module->create<Add>(Node->getType(), Adjoint,
                          Module->createConstantValue(0.f, Node->getType()));
  setAdjoint(Left, LeftAdjoint);
  setAdjoint(Right, RightAdjoint);
}

void AutoDiff::visit(NegPtr Node) {
  auto *Value = Node->getOperand(0);
  auto *Adjoint = getAdjoint(Node);
  auto *AdjointNode = Module->create<Neg>(Node->getType(), Adjoint);
  setAdjoint(Value, AdjointNode);
}

void AutoDiff::visit(DivPtr Node) {
  auto *Upper = Node->getOperand(0);
  auto *Lower = Node->getOperand(1);
  auto *Adjoint = getAdjoint(Node);
  auto *UpperAdjoint = Module->create<Div>(Node->getType(), Adjoint, Lower);
  auto *Square = Module->create<Mul>(Node->getType(), Lower, Lower);
  auto *LowerDiv = Module->create<Div>(Node->getType(), Upper, Square);
  auto *LowerNeg = Module->create<Neg>(Node->getType(), LowerDiv);
  auto *LowerAdjoint = Module->create<Mul>(Node->getType(), Adjoint, LowerNeg);
  setAdjoint(Upper, UpperAdjoint);
  setAdjoint(Lower, LowerAdjoint);
}

void AutoDiff::visit(MulPtr Node) {
  auto *Left = Node->getOperand(0);
  auto *Right = Node->getOperand(1);
  auto *Adjoint = getAdjoint(Node);
  auto *LeftAdjoint = Module->create<Mul>(Node->getType(), Adjoint, Right);
  auto *RightAdjoint = Module->create<Mul>(Node->getType(), Adjoint, Left);
  setAdjoint(Left, LeftAdjoint);
  setAdjoint(Right, RightAdjoint);
}

void AutoDiff::visit(BroadcastPtr Node) {
  auto *Value = Node->getOperand(0);
  auto *Adjoint = getAdjoint(Node);
  auto ValueTensorType = asType<TensorType>(Value->getType());
  auto BroadcastShape = Node->getBroadCastShape();
  std::vector<int64_t> ReduceDims;
  auto Shape = ValueTensorType->getConcreteShape();
  if (Shape.empty()) {
    for (size_t Idx = 0; Idx < BroadcastShape.size(); Idx++) {
      ReduceDims.push_back(Idx);
    }
    auto *AdjointNode =
        Module->create<Sum>(Value->getType(), Adjoint, ReduceDims, false);
    setAdjoint(Value, AdjointNode);
  } else {
    if (Shape.size() == BroadcastShape.size()) {
      for (size_t Idx = 0; Idx < Shape.size(); Idx++) {
        if (Shape[Shape.size() - Idx - 1] !=
            BroadcastShape[Shape.size() - Idx - 1]) {
          ReduceDims.push_back(Shape.size() - Idx - 1);
        }
      }
      auto *AdjointNode =
          Module->create<Sum>(Value->getType(), Adjoint, ReduceDims, true);
      setAdjoint(Value, AdjointNode);
    } else {
      for (size_t Idx = 0; Idx < Shape.size(); Idx++) {
        if (Shape[Shape.size() - Idx - 1] !=
            BroadcastShape[Shape.size() - Idx - 1]) {
          ReduceDims.push_back(Idx);
        }
      }
      auto *AdjointNode =
          Module->create<Sum>(Value->getType(), Adjoint, ReduceDims, false);
      setAdjoint(Value, AdjointNode);
    }
  }
}

void AutoDiff::visit(TransposePtr Node) {
  auto *Value = Node->getValue();
  auto *Adjoint = getAdjoint(Node);
  auto Axes = Node->getAxes();
  auto *AdjointNode = Module->create<Transpose>(Node->getType(), Adjoint, Axes);
  setAdjoint(Value, AdjointNode);
}

void AutoDiff::visit(MatmulPtr Node) {
  auto *Left = Node->getLHS();
  auto *Right = Node->getRHS();
  auto *Adjoint = getAdjoint(Node);
  auto LeftTensorType = asType<TensorType>(Left->getType());
  auto LeftTensorShape = LeftTensorType->getShape();
  std::vector<ValuePtr> LeftTransposeShape;
  for (size_t Idx = 0; Idx < LeftTensorShape.size(); Idx++) {
    LeftTransposeShape.push_back(
        LeftTensorShape[LeftTensorShape.size() - 1 - Idx]);
  }
  auto *LeftTranspose = Module->create<Transpose>(
      TensorType::create(LeftTensorType->getElementType(), LeftTransposeShape),
      Left, std::vector<int>{});
  auto RightTensorType = asType<TensorType>(Right->getType());
  auto RightTensorShape = RightTensorType->getShape();
  std::vector<ValuePtr> RightTransposeShape;
  for (size_t Idx = 0; Idx < RightTensorShape.size(); Idx++) {
    RightTransposeShape.push_back(
        RightTensorShape[RightTensorShape.size() - 1 - Idx]);
  }
  auto *RightTranspose = Module->create<Transpose>(
      TensorType::create(RightTensorType->getElementType(),
                         RightTransposeShape),
      Right, std::vector<int>{});
  auto *LeftAdjoint =
      Module->create<Matmul>(Left->getType(), Adjoint, RightTranspose);
  auto *RightAdjoint =
      Module->create<Matmul>(Right->getType(), LeftTranspose, Adjoint);
  setAdjoint(Left, LeftAdjoint);
  setAdjoint(Right, RightAdjoint);
}
