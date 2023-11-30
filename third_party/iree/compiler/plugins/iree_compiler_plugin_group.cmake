# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(IREE_INPUT_TORCH)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/input/Torch input/Torch)
endif()

if(IREE_TARGET_BACKEND_CUDA)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/target/CUDA target/CUDA)
endif()
