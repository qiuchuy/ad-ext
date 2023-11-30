// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformStrategies/GPU/SmallReductionStrategy.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Strategies.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::IREE::transform_dialect::ForallToWorkgroupOp;
using iree_compiler::IREE::transform_dialect::VectorToWarpExecuteOnLane0Op;
using iree_compiler::IREE::transform_dialect::VectorWarpDistributionOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform::ScalarizeOp;
using transform::SequenceOp;
using transform_ext::MatchCallbackOp;
using transform_ext::RegisterMatchCallbacksOp;
using transform_ext::StructuredOpMatcher;

using iree_compiler::AbstractReductionStrategy;
using iree_compiler::gpu::adjustNumberOfWarpsForBlockShuffle;
using iree_compiler::gpu::build1DSplittingStrategyWithOptionalThreadMapping;
using iree_compiler::gpu::buildCommonTrailingStrategy;
using iree_compiler::gpu::buildDistributeVectors;
using iree_compiler::gpu::kCudaMaxVectorLoadBitWidth;
using iree_compiler::gpu::ReductionConfig;
using iree_compiler::gpu::scaleUpByBitWidth;
using iree_compiler::gpu::SmallReductionStrategy;
using iree_compiler::gpu::threadX;
using iree_compiler::gpu::threadY;
using iree_compiler::gpu::threadZ;

mlir::iree_compiler::gpu::SmallReductionStrategy::SmallReductionStrategy(
    const transform_ext::MatchedReductionCaptures &captures,
    const ReductionConfig &reductionConfig, const GPUModel &gpuModel)
    : AbstractReductionStrategy(captures, {}), GPUStrategy(gpuModel) {
  configure(reductionConfig);
  LLVM_DEBUG(DBGS() << "use GPU small reduction strategy\n");
  LLVM_DEBUG(llvm::interleaveComma(workgroupTileSizes,
                                   DBGS() << "--workgroupTileSizes:  ");
             llvm::dbgs() << "\n");
}

void mlir::iree_compiler::gpu::SmallReductionStrategy::configure(
    const ReductionConfig &reductionConfig) {
  int64_t maxNumThreadsToUse = reductionConfig.maxNumThreads;
  assert(maxNumThreadsToUse > 0 && "maxNumThreadsToUse must be > 0");
  assert(maxNumThreadsToUse >= subgroupSize && "not even a warp?");

  // Block-level
  // ===========
  // TODO: capture more dims than just the most minor parallel and have a more
  // powerful `maybeDivisor` evaluation.
  int64_t mostMinorParallelDimensionSize =
      ArrayRef<int64_t>(captures.reductionOpSizes).drop_back().back();
  FailureOr<int64_t> maybeDivisor = maxDivisorOfValueBelowLimit(
      mostMinorParallelDimensionSize, maxNumThreadsToUse);

  // Trailing elementwise unaligned tiling created bounded local buffers that
  // are dynamic. Attempting to bound them in Common/PadDynamicAlloc.cpp results
  // in a crash in the associated upstream util.
  // TODO: More generally fix PadDynamicAlloc and the associated upstream util.
  bool hasTrailingElementwise = (captures.maybeTrailingRank > 0);
  if (failed(maybeDivisor) && hasTrailingElementwise)
    maybeDivisor = 1;

  // If the captured dimension has no satisfactory divisor, just tile the last
  // parallel dimension by 2 * subgroupSize.
  int64_t numParallelLoops = captures.reductionRank - 1;
  workgroupTileSizes.append(numParallelLoops, 1);
  workgroupTileSizes.back() =
      hasTrailingElementwise
          ? *maybeDivisor
          : std::min((int64_t)maxNumThreadsToUse, (int64_t)(2 * subgroupSize));

  // Thread-level
  // ============
  // Just running sequentially on each thread and relying on cache for
  // locality.
}

static void buildSmallReductionStrategyThreadDistribution(
    ImplicitLocOpBuilder &b, Value variantH, Value maybeLeadingH, Value fillH,
    Value reductionH, Value maybeTrailingH,
    const SmallReductionStrategy &strategy) {
  auto [fusionTargetH, fusionGroupH] =
      iree_compiler::buildSelectFirstNonEmpty(b, maybeTrailingH, reductionH);
  MLIRContext *ctx = b.getContext();
  SmallVector<Attribute> threadDimMapping{threadX(ctx), threadY(ctx),
                                          threadZ(ctx)};
  threadDimMapping.resize(strategy.workgroupTileSizes.size());
  iree_compiler::TileToForallAndFuseAndDistributeResult tileResult =
      iree_compiler::buildTileFuseDistToForallWithNumThreads(
          /*builder=*/b,
          /*variantH=*/variantH,
          /*rootH=*/fusionTargetH,
          /*opsToFuseH=*/fusionGroupH,
          /*numThreads=*/
          getAsOpFoldResult(b.getI64ArrayAttr(strategy.workgroupTileSizes)),
          /*threadDimMapping=*/b.getArrayAttr(threadDimMapping));
  fillH =
      b.create<FuseIntoContainingOp>(fillH, tileResult.forallH).getFusedOp();
  maybeLeadingH =
      b.create<FuseIntoContainingOp>(maybeLeadingH, tileResult.forallH)
          .getFusedOp();

  // 1. Scalarize all ops to ensure vectorization.
  auto anyOpType = transform::AnyOpType::get(b.getContext());
  fillH = b.create<ScalarizeOp>(anyOpType, fillH);
  maybeLeadingH = b.create<ScalarizeOp>(anyOpType, maybeLeadingH);
  Value tiledH = b.create<ScalarizeOp>(anyOpType, tileResult.tiledOpH);
  Value fusedH = b.create<ScalarizeOp>(
      anyOpType, tileResult.resultingFusedOpsHandles.front());
  auto [blockReductionH, maybeBlockTrailingH] =
      iree_compiler::buildSelectFirstNonEmpty(b, fusedH, tiledH);

  // 2. Apply the 1d splitting strategy to the reduction part while specifying
  // a single thread. This triggers the splitting but not the thread mapping
  // part.
  build1DSplittingStrategyWithOptionalThreadMapping(
      /*b=*/b,
      /*variantH=*/variantH,
      /*opH=*/blockReductionH,
      /*rank=*/strategy.captures.reductionRank,
      // TODO: capture and generalize mostMinorDim.
      /*mostMinorDim=*/strategy.captures.reductionRank - 1,
      /*opSizes=*/strategy.captures.reductionOpSizes,
      /*numThreads=*/1);

  // 3. Apply the 1d splitting strategy to the trailing elementwise part while
  // specifying a single thread. This triggers the splitting but not the thread
  // mapping part.
  build1DSplittingStrategyWithOptionalThreadMapping(
      /*b=*/b,
      /*variantH=*/variantH,
      /*opH=*/maybeBlockTrailingH,
      /*rank=*/strategy.captures.maybeTrailingRank,
      // TODO: capture and generalize mostMinorDim.
      /*mostMinorDim=*/strategy.captures.maybeTrailingRank - 1,
      /*opSizes=*/strategy.captures.trailingOpSizes,
      /*numThreads=*/1);
}

void mlir::iree_compiler::gpu::buildSmallReductionStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const SmallReductionStrategy &strategy) {
  // Step 1. Apply block-level part of the strategy, keeps everything fused.
  ArrayRef<int64_t> workgroupTileSizes{strategy.workgroupTileSizes};
  auto [maybeLeadingHBlock, gridFillH, gridReductionH, maybeTiledTrailingHBlock,
        forall] =
      buildReductionStrategyBlockDistribution(
          b, variantH,
          workgroupTileSizes.take_front(strategy.captures.reductionRank - 1));

  // Step 2. Apply thread-level part of the strategy, keeps everything fused.
  buildSmallReductionStrategyThreadDistribution(
      b, variantH, maybeLeadingHBlock, gridFillH, gridReductionH,
      maybeTiledTrailingHBlock, strategy);

  // Step 3-4. Common trailing steps.
  buildCommonTrailingStrategy(b, variantH, strategy.getNumThreadsInBlock());
}
