#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from io import BytesIO, StringIO
import json
import pathlib
import unittest
import zipfile

from common.benchmark_definition import ModuleComponentSizes
from collect_compilation_statistics import (
    CONST_COMPONENT_NAME,
    VM_COMPONENT_NAME,
    get_module_component_info,
    parse_compilation_time_from_ninja_log,
)
from e2e_test_artifacts import iree_artifacts
from e2e_test_framework import serialization
from e2e_test_framework.definitions import common_definitions, iree_definitions
import collect_compilation_statistics
import common.benchmark_definition


class CollectCompilationStatistics(unittest.TestCase):
    def test_match_module_cmake_target_with_e2e_test_artifacts(self):
        target = collect_compilation_statistics.match_module_cmake_target(
            pathlib.PurePath("e2e_test_artifacts/iree_abcd/module.vmfb")
        )

        self.assertEqual(target, "e2e_test_artifacts/iree_abcd/module.vmfb")

    def test_match_module_cmake_target_not_match(self):
        target = collect_compilation_statistics.match_module_cmake_target(
            pathlib.PurePath("other/target.vmfb")
        )

        self.assertIsNone(target)

    def test_parse_compilation_time_from_ninja_log(self):
        target1 = "e2e_test_artifacts/iree_deeplabv3/module.vmfb"
        target2 = "e2e_test_artifacts/iree_mobilessd/module.vmfb"
        ninja_log = StringIO(
            "# ninja log v5\n"
            f"0\t100\taaa\tbuild/{target1}\taaa\n"
            f"130\t200\tbbb\tbuild/{target2}\tbbb\n"
        )

        target_map = parse_compilation_time_from_ninja_log(ninja_log)

        self.assertEqual(target_map, {target1: 100, target2: 70})

    def test_get_module_component_info(self):
        module_file = BytesIO()
        with zipfile.ZipFile(module_file, "w") as zip:
            zip.writestr(VM_COMPONENT_NAME, b"abcd")
            zip.writestr(CONST_COMPONENT_NAME, b"123")
            zip.writestr("main_dispatch_0_vulkan_spirv_fb.fb", b"bindata0")
            zip.writestr("main_dispatch_1_vulkan_spirv_fb.fb", b"bindata1")
            zip.writestr("predict_dispatch_2_cuda_nvptx_fb.fb", b"bindata2")
            zip.writestr("dispatch_3_embedded_elf_x86_64.so", b"bindata3")
        module_file_data = module_file.getvalue()

        component_sizes = get_module_component_info(
            BytesIO(module_file_data), len(module_file_data)
        )

        self.assertEqual(
            component_sizes,
            ModuleComponentSizes(
                file_bytes=len(module_file_data),
                vm_component_bytes=4,
                const_component_bytes=3,
                total_dispatch_component_bytes=32,
            ),
        )

    def test_get_module_map_from_compilation_benchmark_config(self):
        model_a = common_definitions.Model(
            id="1234",
            name="tflite_m",
            tags=[],
            source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
            source_url="https://example.com/xyz.tflite",
            entry_function="main",
            input_types=["1xf32"],
        )
        imported_model_a = iree_definitions.ImportedModel.from_model(model_a)
        compile_config_a = iree_definitions.CompileConfig.build(
            id="config_a",
            tags=["defaults"],
            compile_targets=[
                iree_definitions.CompileTarget(
                    target_architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
                    target_backend=iree_definitions.TargetBackend.LLVM_CPU,
                    target_abi=iree_definitions.TargetABI.LINUX_GNU,
                )
            ],
        )
        compile_config_b = iree_definitions.CompileConfig.build(
            id="config_b",
            tags=["defaults"],
            compile_targets=[
                iree_definitions.CompileTarget(
                    target_architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
                    target_backend=iree_definitions.TargetBackend.LLVM_CPU,
                    target_abi=iree_definitions.TargetABI.LINUX_GNU,
                )
            ],
        )
        gen_config_a = iree_definitions.ModuleGenerationConfig.build(
            imported_model=imported_model_a, compile_config=compile_config_a
        )
        gen_config_b = iree_definitions.ModuleGenerationConfig.build(
            imported_model=imported_model_a, compile_config=compile_config_b
        )
        benchmark_config = dict(
            generation_configs=serialization.serialize_and_pack(
                [gen_config_a, gen_config_b]
            ),
            module_dir_paths=["a", "b"],
        )
        root_dir = pathlib.PurePath("artifacts_dir")

        module_map = collect_compilation_statistics.get_module_map_from_compilation_benchmark_config(
            compilation_benchmark_config_data=StringIO(json.dumps(benchmark_config)),
            e2e_test_artifacts_dir=root_dir,
        )

        compile_info_a = common.benchmark_definition.CompilationInfo(
            name=gen_config_a.name,
            model_name=model_a.name,
            model_tags=tuple(model_a.tags),
            model_source=model_a.source_type.value,
            target_arch=f"[cpu-x86_64-cascadelake-linux-gnu]",
            compile_tags=tuple(gen_config_a.compile_config.tags),
            gen_config_id=gen_config_a.composite_id,
        )
        module_dir_a = pathlib.Path(
            iree_artifacts.get_module_dir_path(gen_config_a, root_dir)
        )
        module_info_a = collect_compilation_statistics.ModuleInfo(
            module_path=module_dir_a / iree_artifacts.MODULE_FILENAME,
            stream_stats_path=module_dir_a / iree_artifacts.SCHEDULING_STATS_FILENAME,
        )
        compile_info_b = common.benchmark_definition.CompilationInfo(
            name=gen_config_b.name,
            model_name=model_a.name,
            model_tags=tuple(model_a.tags),
            model_source=model_a.source_type.value,
            target_arch=f"[cpu-riscv_64-generic-linux-gnu]",
            compile_tags=tuple(gen_config_a.compile_config.tags),
            gen_config_id=gen_config_b.composite_id,
        )
        module_dir_b = pathlib.Path(
            iree_artifacts.get_module_dir_path(gen_config_b, root_dir)
        )
        module_info_b = collect_compilation_statistics.ModuleInfo(
            module_path=module_dir_b / iree_artifacts.MODULE_FILENAME,
            stream_stats_path=module_dir_b / iree_artifacts.SCHEDULING_STATS_FILENAME,
        )
        self.assertEqual(
            module_map, {compile_info_a: module_info_a, compile_info_b: module_info_b}
        )


if __name__ == "__main__":
    unittest.main()
