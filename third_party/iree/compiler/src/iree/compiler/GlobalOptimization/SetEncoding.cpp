// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- SetEncoding.cpp -------------------------------------===//
// Sets the encoding for compute operations to allow execution of the
// operations in tiled layouts.
//===---------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

//===---------------------------------------------------------------------===//
// Utility functions
//===---------------------------------------------------------------------===//

/// Pads `value` enough for any actual tile sizes that could result from
/// materialization of `encodingAttr`.
static Value pad(OpBuilder &builder, Location loc, Value source,
                 IREE::LinalgExt::EncodingAttr encodingAttr) {
  RankedTensorType sourceType = source.getType().cast<RankedTensorType>();
  Type elemType = sourceType.getElementType();
  size_t rank = sourceType.getRank();
  RankedTensorType tensorTypeWithEncoding =
      RankedTensorType::get(sourceType.getShape(), elemType, encodingAttr);
  SmallVector<OpFoldResult> lowPad(rank, builder.getIndexAttr(0));
  SmallVector<Type> resultTypes(rank, builder.getIndexType());

  ValueRange encodingPaddingSizes =
      builder
          .create<IREE::LinalgExt::UpperBoundTileSizeOp>(
              loc, resultTypes, TypeAttr::get(tensorTypeWithEncoding))
          .getResults();
  SmallVector<OpFoldResult> highPad(rank);
  AffineExpr tileExpr, shapeExpr;
  bindSymbols(builder.getContext(), tileExpr, shapeExpr);
  AffineExpr highPadExpr = shapeExpr.ceilDiv(tileExpr) * tileExpr - shapeExpr;
  for (size_t i = 0; i < rank; ++i) {
    highPad[i] = affine::makeComposedFoldedAffineApply(
        builder, loc, highPadExpr,
        getAsOpFoldResult({encodingPaddingSizes[i],
                           builder.create<tensor::DimOp>(loc, source, i)}));
  }

  Value zero = builder.create<arith::ConstantOp>(loc, elemType,
                                                 builder.getZeroAttr(elemType));
  return builder.create<tensor::PadOp>(loc, /*resultType=*/nullptr, source,
                                       lowPad, highPad, zero);
}

static Value setEncoding(OpBuilder &builder, Location loc, Value source,
                         IREE::LinalgExt::EncodingAttr encodingAttr) {
  auto sourceType = source.getType().cast<RankedTensorType>();
  auto resultType = RankedTensorType::get(
      sourceType.getShape(), sourceType.getElementType(), encodingAttr);
  return builder.create<IREE::LinalgExt::SetEncodingOp>(loc, resultType,
                                                        source);
};

static IREE::LinalgExt::EncodingAttr
makeEncoding(OpBuilder &builder, IREE::LinalgExt::EncodingUser user,
             IREE::LinalgExt::EncodingRole role, TypeRange operandTypes,
             Type originalType) {
  auto *context = builder.getContext();
  auto userAttr = IREE::LinalgExt::EncodingUserAttr::get(context, user);
  auto roleAttr = IREE::LinalgExt::EncodingRoleAttr::get(context, role);
  SmallVector<Attribute> elemTypeAttrs =
      llvm::map_to_vector(operandTypes, [](auto t) {
        return TypeAttr::get(t.template cast<ShapedType>().getElementType())
            .template cast<Attribute>();
      });
  auto operandElemTypesAttr = ArrayAttr::get(context, elemTypeAttrs);
  auto originalTypeAttr =
      originalType ? TypeAttr::get(originalType) : TypeAttr{};
  return IREE::LinalgExt::EncodingAttr::get(
      context, userAttr, roleAttr, operandElemTypesAttr, originalTypeAttr);
}

static Value padAndSetEncoding(OpBuilder &builder, Location loc, Value source,
                               IREE::LinalgExt::EncodingUser user,
                               IREE::LinalgExt::EncodingRole role,
                               TypeRange operandTypes) {
  // No need to specify original_type in the encoding poadded to pad(), because
  // the operand there is the `source` tensor, so it will default to reading its
  // original shape.
  auto encodingForPad =
      makeEncoding(builder, user, role, operandTypes, /*originalType=*/Type{});
  Value padded = pad(builder, loc, source, encodingForPad);
  // For setEncoding() below, we potentially need to specify an encoding with an
  // explicit original_type, because the operand there is the padded tensor
  // returned by pad() above, but we want setEncoding to be aware of the
  // original source tensor shape, not the padded tensor shape. To limit IR
  // verbosity, we only specify the original original_type when it differs from
  // the tensor type that the encoding is applied to.
  auto encodingForSetEncoding = encodingForPad;
  if (padded.getType() != source.getType()) {
    encodingForSetEncoding =
        makeEncoding(builder, user, role, operandTypes, source.getType());
  }
  return setEncoding(builder, loc, padded, encodingForSetEncoding);
}

static Value unsetEncodingAndExtractSlice(OpBuilder &builder, Location loc,
                                          Value source,
                                          SmallVector<OpFoldResult> sizes) {
  auto sourceType = source.getType().cast<RankedTensorType>();
  auto unsetEncodingReturnType =
      RankedTensorType::get(sourceType.getShape(), sourceType.getElementType());
  auto unsetEncoding = builder
                           .create<IREE::LinalgExt::UnsetEncodingOp>(
                               loc, unsetEncodingReturnType, source)
                           .getResult();
  auto rank = sourceType.getRank();
  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  return builder.create<tensor::ExtractSliceOp>(loc, unsetEncoding, offsets,
                                                sizes, strides);
}

namespace {

/// Rewrites the matmul op to work on tensors with encoding. Optionally
/// also pads the operands.
struct SetMatmulEncoding : public OpRewritePattern<linalg::MatmulOp> {
  SetMatmulEncoding(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasTensorSemantics())
      return failure();
    auto inputs = matmulOp.getDpsInputs();
    auto outputs = matmulOp.getDpsInits();
    auto hasEncoding = [](Value operand) -> bool {
      auto type = llvm::dyn_cast<RankedTensorType>(operand.getType());
      return type && type.getEncoding();
    };
    if (llvm::any_of(inputs, hasEncoding) ||
        llvm::any_of(outputs, hasEncoding)) {
      return failure();
    }

    Value origLhs = inputs[0];
    Value origRhs = inputs[1];
    Value origOut = outputs[0];

    auto getElemType = [](Value v) -> Type {
      if (auto tensorType = llvm::dyn_cast<RankedTensorType>(v.getType())) {
        return tensorType.getElementType();
      }
      return {};
    };
    Type lhsElemType = getElemType(origLhs);
    Type rhsElemType = getElemType(origRhs);
    Type outElemType = getElemType(origOut);

    if (!lhsElemType || !rhsElemType || !outElemType) {
      return failure();
    }

    IREE::LinalgExt::EncodingUser user = IREE::LinalgExt::EncodingUser::MATMUL;
    Location loc = matmulOp.getLoc();
    TypeRange operandTypes = matmulOp->getOperandTypes();
    Value encodedLhs =
        padAndSetEncoding(rewriter, loc, origLhs, user,
                          IREE::LinalgExt::EncodingRole::LHS, operandTypes);
    Value encodedRhs =
        padAndSetEncoding(rewriter, loc, origRhs, user,
                          IREE::LinalgExt::EncodingRole::RHS, operandTypes);
    Value encodedOut =
        padAndSetEncoding(rewriter, loc, origOut, user,
                          IREE::LinalgExt::EncodingRole::RESULT, operandTypes);

    Value matmulTiled = rewriter
                            .create<linalg::MatmulOp>(
                                loc, encodedOut.getType(),
                                ValueRange{encodedLhs, encodedRhs}, encodedOut)
                            .getResult(0);

    // Sizes are computed by original output size.
    FailureOr<SmallVector<OpFoldResult>> origOutSizes =
        IREE::LinalgExt::getDims(rewriter, loc, origOut);
    if (failed(origOutSizes)) {
      return rewriter.notifyMatchFailure(matmulOp,
                                         "failed to get shape of result");
    }

    Value result = unsetEncodingAndExtractSlice(rewriter, loc, matmulTiled,
                                                origOutSizes.value());

    rewriter.replaceOp(matmulOp, result);
    return success();
  }
};

struct SetBatchMatmulEncoding : public OpRewritePattern<linalg::BatchMatmulOp> {
  SetBatchMatmulEncoding(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::BatchMatmulOp>(context, benefit) {}

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasTensorSemantics())
      return failure();
    auto inputs = matmulOp.getDpsInputs();
    auto outputs = matmulOp.getDpsInits();
    auto hasEncoding = [](Value operand) -> bool {
      auto type = llvm::dyn_cast<RankedTensorType>(operand.getType());
      return type && type.getEncoding();
    };
    if (llvm::any_of(inputs, hasEncoding) ||
        llvm::any_of(outputs, hasEncoding)) {
      return failure();
    }

    Value origLhs = inputs[0];
    Value origRhs = inputs[1];
    Value origOut = outputs[0];

    auto getElemType = [](Value v) -> Type {
      if (auto tensorType = llvm::dyn_cast<RankedTensorType>(v.getType())) {
        return tensorType.getElementType();
      }
      return {};
    };
    Type lhsElemType = getElemType(origLhs);
    Type rhsElemType = getElemType(origRhs);
    Type outElemType = getElemType(origOut);

    if (!lhsElemType || !rhsElemType || !outElemType) {
      return failure();
    }

    IREE::LinalgExt::EncodingUser user =
        IREE::LinalgExt::EncodingUser::BATCH_MATMUL;
    Location loc = matmulOp.getLoc();
    TypeRange operandTypes = matmulOp->getOperandTypes();
    Value encodedLhs =
        padAndSetEncoding(rewriter, loc, origLhs, user,
                          IREE::LinalgExt::EncodingRole::LHS, operandTypes);
    Value encodedRhs =
        padAndSetEncoding(rewriter, loc, origRhs, user,
                          IREE::LinalgExt::EncodingRole::RHS, operandTypes);
    Value encodedOut =
        padAndSetEncoding(rewriter, loc, origOut, user,
                          IREE::LinalgExt::EncodingRole::RESULT, operandTypes);

    Value matmulTiled = rewriter
                            .create<linalg::BatchMatmulOp>(
                                loc, encodedOut.getType(),
                                ValueRange{encodedLhs, encodedRhs}, encodedOut)
                            .getResult(0);

    // Sizes are computed by original output size.
    FailureOr<SmallVector<OpFoldResult>> origOutSizes =
        IREE::LinalgExt::getDims(rewriter, loc, origOut);
    if (failed(origOutSizes)) {
      return rewriter.notifyMatchFailure(matmulOp,
                                         "failed to get shape of result");
    }

    Value result = unsetEncodingAndExtractSlice(rewriter, loc, matmulTiled,
                                                origOutSizes.value());

    rewriter.replaceOp(matmulOp, result);
    return success();
  }
};

/// Pattern to fold a `linalg.fill` -> `iree_linalg_ext.set_encoding`
/// operation into a `linalg.fill` of the encoded type.
struct FoldFillWithSetEncoding
    : public OpRewritePattern<IREE::LinalgExt::SetEncodingOp> {
  using OpRewritePattern<IREE::LinalgExt::SetEncodingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::SetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = encodingOp.getSource().getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return failure();

    // Create a new fill op, with outs being defined by a new `tensor.empty` op.
    RankedTensorType encodingType = encodingOp.getResultType();
    Location loc = fillOp.getLoc();
    SmallVector<OpFoldResult> dimValues =
        tensor::getMixedSizes(rewriter, loc, fillOp.getOutputs()[0]);
    auto newEmptyOp = rewriter.create<tensor::EmptyOp>(
        loc, dimValues, encodingType.getElementType(),
        encodingType.getEncoding());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(encodingOp, fillOp.getInputs(),
                                                ValueRange{newEmptyOp});
    return success();
  }
};

struct SetEncodingPass : public SetEncodingBase<SetEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void SetEncodingPass::runOnOperation() {
  MLIRContext *context = &getContext();
  {
    RewritePatternSet patterns(context);
    patterns.insert<SetBatchMatmulEncoding, SetMatmulEncoding>(context);
    linalg::FillOp::getCanonicalizationPatterns(patterns, context);
    patterns.insert<FoldFillWithSetEncoding>(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<Pass> createSetEncodingPass() {
  return std::make_unique<SetEncodingPass>();
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
