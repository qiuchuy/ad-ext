// RUN: iree-opt --iree-global-opt-expand-vectors --split-input-file %s | FileCheck %s

func.func @vecmat_f32f32f32(%arg0 : tensor<250xf32>, %arg1 : tensor<250x100xf32>,
    %arg2 : tensor<100xf32>) -> tensor<100xf32> {
  %0 = linalg.vecmat ins(%arg0, %arg1 : tensor<250xf32>, tensor<250x100xf32>)
      outs(%arg2 : tensor<100xf32>) -> tensor<100xf32>
  return %0 : tensor<100xf32>
}
//      CHECK:  func @vecmat_f32f32f32(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<250xf32>, %[[ARG1:.+]]: tensor<250x100xf32>, %[[ARG2:.+]]: tensor<100xf32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<250xf32> into tensor<1x250xf32>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1]] : tensor<100xf32> into tensor<1x100xf32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.matmul ins(%[[EXPANDED_IN]], %[[ARG1]] : tensor<1x250xf32>, tensor<250x100xf32>) outs(%[[EXPANDED_OUT]] : tensor<1x100xf32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0, 1]] : tensor<1x100xf32> into tensor<100xf32>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @vecmat_bf16bf16f32_dynamic(%arg0 : tensor<?xbf16>, %arg1 : tensor<?x?xbf16>,
    %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.vecmat ins(%arg0, %arg1 : tensor<?xbf16>, tensor<?x?xbf16>)
      outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
//      CHECK:  func @vecmat_bf16bf16f32_dynamic(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<?xbf16>, %[[ARG1:.+]]: tensor<?x?xbf16>, %[[ARG2:.+]]: tensor<?xf32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<?xbf16> into tensor<1x?xbf16>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1]] : tensor<?xf32> into tensor<1x?xf32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.matmul ins(%[[EXPANDED_IN]], %[[ARG1]] : tensor<1x?xbf16>, tensor<?x?xbf16>) outs(%[[EXPANDED_OUT]] : tensor<1x?xf32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @matvec_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250xf32>,
    %arg2 : tensor<100xf32>) -> tensor<100xf32> {
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250xf32>)
      outs(%arg2 : tensor<100xf32>) -> tensor<100xf32>
  return %0 : tensor<100xf32>
}
//      CHECK:  func @matvec_f32f32f32(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<100x250xf32>, %[[ARG1:.+]]: tensor<250xf32>, %[[ARG2:.+]]: tensor<100xf32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1]] : tensor<250xf32> into tensor<250x1xf32>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1]] : tensor<100xf32> into tensor<100x1xf32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.matmul ins(%[[ARG0]], %[[EXPANDED_IN]] : tensor<100x250xf32>, tensor<250x1xf32>) outs(%[[EXPANDED_OUT]] : tensor<100x1xf32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0, 1]] : tensor<100x1xf32> into tensor<100xf32>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @matvec_i8i8i32_dynamic(%arg0 : tensor<?x?xi8>, %arg1 : tensor<?xi8>,
    %arg2 : tensor<?xi32>) -> tensor<?xi32> {
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<?x?xi8>, tensor<?xi8>)
      outs(%arg2 : tensor<?xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
//      CHECK:  func @matvec_i8i8i32_dynamic(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<?x?xi8>, %[[ARG1:.+]]: tensor<?xi8>, %[[ARG2:.+]]: tensor<?xi32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1]] : tensor<?xi8> into tensor<?x1xi8>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1]] : tensor<?xi32> into tensor<?x1xi32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.matmul ins(%[[ARG0]], %[[EXPANDED_IN]] : tensor<?x?xi8>, tensor<?x1xi8>) outs(%[[EXPANDED_OUT]] : tensor<?x1xi32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0, 1]] : tensor<?x1xi32> into tensor<?xi32>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @batch_matvec_f32f32f32(%arg0 : tensor<3x100x250xf32>, %arg1 : tensor<3x250xf32>,
    %arg2 : tensor<3x100xf32>) -> tensor<3x100xf32> {
  %0 = linalg.batch_matvec ins(%arg0, %arg1 : tensor<3x100x250xf32>, tensor<3x250xf32>)
      outs(%arg2 : tensor<3x100xf32>) -> tensor<3x100xf32>
  return %0 : tensor<3x100xf32>
}
//      CHECK:  func @batch_matvec_f32f32f32(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<3x100x250xf32>, %[[ARG1:.+]]: tensor<3x250xf32>, %[[ARG2:.+]]: tensor<3x100xf32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0], [1, 2]] : tensor<3x250xf32> into tensor<3x250x1xf32>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0], [1, 2]] : tensor<3x100xf32> into tensor<3x100x1xf32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.batch_matmul ins(%[[ARG0]], %[[EXPANDED_IN]] : tensor<3x100x250xf32>, tensor<3x250x1xf32>) outs(%[[EXPANDED_OUT]] : tensor<3x100x1xf32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0], [1, 2]] : tensor<3x100x1xf32> into tensor<3x100xf32>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @batch_matvec_f16f16f16_dynamic(%arg0 : tensor<?x?x?xf16>, %arg1 : tensor<?x?xf16>,
    %arg2 : tensor<?x?xf16>) -> tensor<?x?xf16> {
  %0 = linalg.batch_matvec ins(%arg0, %arg1 : tensor<?x?x?xf16>, tensor<?x?xf16>)
      outs(%arg2 : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}
//      CHECK:  func @batch_matvec_f16f16f16_dynamic(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<?x?x?xf16>, %[[ARG1:.+]]: tensor<?x?xf16>, %[[ARG2:.+]]: tensor<?x?xf16>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0], [1, 2]] : tensor<?x?xf16> into tensor<?x?x1xf16>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0], [1, 2]] : tensor<?x?xf16> into tensor<?x?x1xf16>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.batch_matmul ins(%[[ARG0]], %[[EXPANDED_IN]] : tensor<?x?x?xf16>, tensor<?x?x1xf16>) outs(%[[EXPANDED_OUT]] : tensor<?x?x1xf16>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0], [1, 2]] : tensor<?x?x1xf16> into tensor<?x?xf16>
//      CHECK:  return %[[COLLAPSED]]
