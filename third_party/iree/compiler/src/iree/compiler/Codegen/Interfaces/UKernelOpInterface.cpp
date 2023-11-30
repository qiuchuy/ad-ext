// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

// clang-format off
#include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.cpp.inc" // IWYU pragma: keep
// clang-format on
