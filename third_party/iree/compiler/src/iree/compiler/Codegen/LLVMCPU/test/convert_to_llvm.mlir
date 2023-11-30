// RUN: iree-opt -iree-convert-to-llvm --split-input-file %s | FileCheck %s

builtin.module {
  func.func private @extern_public()
  func.func @entry_point() {
    return
  }
}
//      CHECK: llvm.func @extern_public()
//      CHECK: llvm.func @entry_point(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias},
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias},
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias}) -> i32
//      CHECK:     llvm.return %{{.+}} : i32

// -----

module {
  func.func private @default_cconv_with_extra_fields(memref<f32>, i32, f64) -> (f32) attributes {
      hal.import.bitcode = true,
      hal.import.cconv = 0 : i32,
      hal.import.fields = ["processor_data", "processor_id"],
      llvm.bareptr = true
  }
  func.func @bar() {
    %c0 = arith.constant 42 : i32
    %c1 = arith.constant 42.0 : f64
    %0 = memref.alloca() : memref<f32>
    %1 = call @default_cconv_with_extra_fields(%0, %c0, %c1) : (memref<f32>, i32, f64) -> (f32)
    return
  }
}
//      CHECK: llvm.func @default_cconv_with_extra_fields(!llvm.ptr, i32, f64, !llvm.ptr, i32) -> f32
//      CHECK: llvm.func @bar
//  CHECK-DAG:   %[[Ci32:.+]] = llvm.mlir.constant(42 : i32) : i32
//  CHECK-DAG:   %[[Cf64:.+]] = llvm.mlir.constant(4.200000e+01 : f64) : f64
//  CHECK-DAG:   %[[ALLOCA:.+]] = llvm.alloca
//  CHECK-DAG:   %[[DATA:.+]] = llvm.getelementptr inbounds %arg0[4]
//  CHECK-DAG:   %[[PROCESSOR_INFO:.+]] = llvm.load %arg2
//      CHECK:   %[[PROCESSOR_ID:.+]] = llvm.extractvalue %[[PROCESSOR_INFO]][4]
//      CHECK: %[[VAL:.+]] = llvm.call @default_cconv_with_extra_fields
// CHECK-SAME:     (%[[ALLOCA]], %[[Ci32]], %[[Cf64]], %[[DATA]], %[[PROCESSOR_ID]])

// -----

module {
  func.func private @paramstruct_cconv_with_extra_fields(memref<f32>, i32, f64) -> (f32) attributes {
      hal.import.bitcode = true,
      hal.import.cconv = 1 : i32,
      hal.import.fields = ["processor_data", "processor_id"],
      llvm.bareptr = true
  }
  func.func @bar() {
    %c0 = arith.constant 42 : i32
    %c1 = arith.constant 42.0 : f64
    %0 = memref.alloca() : memref<f32>
    %1 = call @paramstruct_cconv_with_extra_fields(%0, %c0, %c1) : (memref<f32>, i32, f64) -> (f32)
    return
  }
}
//      CHECK: llvm.func @paramstruct_cconv_with_extra_fields(!llvm.ptr)
//      CHECK: llvm.func @bar
//  CHECK-DAG:   %[[C1:.+]] = llvm.mlir.constant(1 : index) : i64
//  CHECK-DAG:   %[[Ci32:.+]] = llvm.mlir.constant(42 : i32) : i32
//  CHECK-DAG:   %[[Cf64:.+]] = llvm.mlir.constant(4.200000e+01 : f64) : f64
//  CHECK-DAG:   %[[ALLOCA:.+]] = llvm.alloca
//  CHECK-DAG:   %[[DATA:.+]] = llvm.getelementptr inbounds %arg0[4]
//  CHECK-DAG:   %[[PROCESSOR_INFO:.+]] = llvm.load %arg2
//      CHECK:   %[[PROCESSOR_ID:.+]] = llvm.extractvalue %[[PROCESSOR_INFO]][4]
//      CHECK:   %[[PARAMSTRUCT_ALLOCA:.+]] = llvm.alloca %[[C1]] x !llvm.struct<(f32, ptr, i32, f64, ptr, i32)>
//      CHECK:   %[[PARAMSTRUCT:.+]] = llvm.mlir.undef : !llvm.struct<(f32, ptr, i32, f64, ptr, i32)>
//      CHECK:   %[[INSERT_ARG0:.+]] = llvm.insertvalue %[[ALLOCA]], %[[PARAMSTRUCT]][1]
//      CHECK:   %[[INSERT_ARG1:.+]] = llvm.insertvalue %[[Ci32]], %[[INSERT_ARG0]][2]
//      CHECK:   %[[INSERT_ARG2:.+]] = llvm.insertvalue %[[Cf64]], %[[INSERT_ARG1]][3]
//      CHECK:   %[[INSERT_ARG3:.+]] = llvm.insertvalue %[[DATA]], %[[INSERT_ARG2]][4]
//      CHECK:   %[[INSERT_ARG4:.+]] = llvm.insertvalue %[[PROCESSOR_ID]], %[[INSERT_ARG3]][5]
//      CHECK:   llvm.store %[[INSERT_ARG4]], %[[PARAMSTRUCT_ALLOCA]]
//      CHECK:   llvm.call @paramstruct_cconv_with_extra_fields(%[[PARAMSTRUCT_ALLOCA]])

// -----

module attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
      cpu = "haswell",
      cpu_features = "+mmx,+popcnt,+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+avx,+avx2,+fma,+bmi,+bmi2,+pclmul,+cx16,+cx8,+crc32,+f16c,+fsgsbase,+fxsr,+invpcid,+lzcnt,+movbe,+rdrnd,+sahf,+x87,+xsave,+xsaveopt",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 32 : index,
      target_triple = "x86_64-none-elf", ukernels = true}>} {
  func.func private @paramstruct_cconv_with_extra_fields_and_executable_target(memref<f32>, i32, f64) -> (f32) attributes {
      hal.import.bitcode = true,
      hal.import.cconv = 1 : i32,
      hal.import.fields = ["processor_data", "processor_id"],
      llvm.bareptr = true
  }
  func.func @bar() {
    %c0 = arith.constant 42 : i32
    %c1 = arith.constant 42.0 : f64
    %0 = memref.alloca() : memref<f32>
    %1 = call @paramstruct_cconv_with_extra_fields_and_executable_target(%0, %c0, %c1) : (memref<f32>, i32, f64) -> (f32)
    return
  }
}
//      CHECK: llvm.func @paramstruct_cconv_with_extra_fields_and_executable_target(!llvm.ptr)
//      CHECK: llvm.func @bar
//  CHECK-DAG:   %[[CPUDATA_FIELD0:.+]] = llvm.mlir.constant(52239 : i64) : i64
//  CHECK-DAG:   %[[C8:.+]] = llvm.mlir.constant(8 : i64) : i64
//  CHECK-DAG:   %[[C1:.+]] = llvm.mlir.constant(1 : index) : i64
//  CHECK-DAG:   %[[Ci32:.+]] = llvm.mlir.constant(42 : i32) : i32
//  CHECK-DAG:   %[[Cf64:.+]] = llvm.mlir.constant(4.200000e+01 : f64) : f64
//  CHECK-DAG:   %[[ALLOCA:.+]] = llvm.alloca
//  CHECK-DAG:   %[[DATA_PTR:.+]] = llvm.getelementptr inbounds %arg0[4]
//      CHECK:   %[[PROCESSOR_DATA_ALLOCA:.+]] = llvm.alloca %[[C8]] x i64 {alignment = 8 : i64}
//  CHECK-DAG:   %[[DATA:.+]] = llvm.load %[[DATA_PTR]]
//  CHECK-DAG:   %[[OR0:.+]] = llvm.or %[[DATA]], %[[CPUDATA_FIELD0]]
//      CHECK:   llvm.store %[[OR0]], %[[PROCESSOR_DATA_ALLOCA]]

//      CHECK:   %[[PROCESSOR_DATA_PTR_1:.+]] = llvm.getelementptr inbounds %[[DATA_PTR]][1]
//      CHECK:   %[[PROCESSOR_DATA_1:.+]] = llvm.load %[[PROCESSOR_DATA_PTR_1]]
//      CHECK:   %[[PROCESSOR_DATA_ALLOCA_PTR_1:.+]] = llvm.getelementptr inbounds %[[PROCESSOR_DATA_ALLOCA]][1]
//      CHECK:   llvm.store %[[PROCESSOR_DATA_1]], %[[PROCESSOR_DATA_ALLOCA_PTR_1]]

//      CHECK:   %[[PROCESSOR_DATA_PTR_2:.+]] = llvm.getelementptr inbounds %[[DATA_PTR]][2]
//      CHECK:   %[[PROCESSOR_DATA_2:.+]] = llvm.load %[[PROCESSOR_DATA_PTR_2]]
//      CHECK:   %[[PROCESSOR_DATA_ALLOCA_PTR_2:.+]] = llvm.getelementptr inbounds %[[PROCESSOR_DATA_ALLOCA]][2]
//      CHECK:   llvm.store %[[PROCESSOR_DATA_2]], %[[PROCESSOR_DATA_ALLOCA_PTR_2]]

//      CHECK:   %[[PROCESSOR_DATA_PTR_3:.+]] = llvm.getelementptr inbounds %[[DATA_PTR]][3]
//      CHECK:   %[[PROCESSOR_DATA_3:.+]] = llvm.load %[[PROCESSOR_DATA_PTR_3]]
//      CHECK:   %[[PROCESSOR_DATA_ALLOCA_PTR_3:.+]] = llvm.getelementptr inbounds %[[PROCESSOR_DATA_ALLOCA]][3]
//      CHECK:   llvm.store %[[PROCESSOR_DATA_3]], %[[PROCESSOR_DATA_ALLOCA_PTR_3]]

//      CHECK:   %[[PROCESSOR_DATA_PTR_4:.+]] = llvm.getelementptr inbounds %[[DATA_PTR]][4]
//      CHECK:   %[[PROCESSOR_DATA_4:.+]] = llvm.load %[[PROCESSOR_DATA_PTR_4]]
//      CHECK:   %[[PROCESSOR_DATA_ALLOCA_PTR_4:.+]] = llvm.getelementptr inbounds %[[PROCESSOR_DATA_ALLOCA]][4]
//      CHECK:   llvm.store %[[PROCESSOR_DATA_4]], %[[PROCESSOR_DATA_ALLOCA_PTR_4]]

//      CHECK:   %[[PROCESSOR_DATA_PTR_5:.+]] = llvm.getelementptr inbounds %[[DATA_PTR]][5]
//      CHECK:   %[[PROCESSOR_DATA_5:.+]] = llvm.load %[[PROCESSOR_DATA_PTR_5]]
//      CHECK:   %[[PROCESSOR_DATA_ALLOCA_PTR_5:.+]] = llvm.getelementptr inbounds %[[PROCESSOR_DATA_ALLOCA]][5]
//      CHECK:   llvm.store %[[PROCESSOR_DATA_5]], %[[PROCESSOR_DATA_ALLOCA_PTR_5]]

//      CHECK:   %[[PROCESSOR_DATA_PTR_6:.+]] = llvm.getelementptr inbounds %[[DATA_PTR]][6]
//      CHECK:   %[[PROCESSOR_DATA_6:.+]] = llvm.load %[[PROCESSOR_DATA_PTR_6]]
//      CHECK:   %[[PROCESSOR_DATA_ALLOCA_PTR_6:.+]] = llvm.getelementptr inbounds %[[PROCESSOR_DATA_ALLOCA]][6]
//      CHECK:   llvm.store %[[PROCESSOR_DATA_6]], %[[PROCESSOR_DATA_ALLOCA_PTR_6]]

//      CHECK:   %[[PROCESSOR_DATA_PTR_7:.+]] = llvm.getelementptr inbounds %[[DATA_PTR]][7]
//      CHECK:   %[[PROCESSOR_DATA_7:.+]] = llvm.load %[[PROCESSOR_DATA_PTR_7]]
//      CHECK:   %[[PROCESSOR_DATA_ALLOCA_PTR_7:.+]] = llvm.getelementptr inbounds %[[PROCESSOR_DATA_ALLOCA]][7]
//      CHECK:   llvm.store %[[PROCESSOR_DATA_7]], %[[PROCESSOR_DATA_ALLOCA_PTR_7]]

//  CHECK-DAG:   %[[PROCESSOR_INFO:.+]] = llvm.load %arg2
//      CHECK:   %[[PROCESSOR_ID:.+]] = llvm.extractvalue %[[PROCESSOR_INFO]][4]
//      CHECK:   %[[PARAMSTRUCT_ALLOCA:.+]] = llvm.alloca %[[C1]] x !llvm.struct<(f32, ptr, i32, f64, ptr, i32)>
//      CHECK:   %[[PARAMSTRUCT:.+]] = llvm.mlir.undef : !llvm.struct<(f32, ptr, i32, f64, ptr, i32)>
//      CHECK:   %[[INSERT_ARG0:.+]] = llvm.insertvalue %[[ALLOCA]], %[[PARAMSTRUCT]][1]
//      CHECK:   %[[INSERT_ARG1:.+]] = llvm.insertvalue %[[Ci32]], %[[INSERT_ARG0]][2]
//      CHECK:   %[[INSERT_ARG2:.+]] = llvm.insertvalue %[[Cf64]], %[[INSERT_ARG1]][3]
//      CHECK:   %[[INSERT_ARG3:.+]] = llvm.insertvalue %[[PROCESSOR_DATA_ALLOCA]], %[[INSERT_ARG2]][4]
//      CHECK:   %[[INSERT_ARG4:.+]] = llvm.insertvalue %[[PROCESSOR_ID]], %[[INSERT_ARG3]][5]
//      CHECK:   llvm.store %[[INSERT_ARG4]], %[[PARAMSTRUCT_ALLOCA]]
//      CHECK:   llvm.call @paramstruct_cconv_with_extra_fields_and_executable_target(%[[PARAMSTRUCT_ALLOCA]])
