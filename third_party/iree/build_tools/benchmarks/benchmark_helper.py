#!/usr/bin/env python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Miscellaneous tool to help work with benchmark suite and benchmark CI."""

import pathlib
import sys

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

import argparse
import json
import os
import shlex
import subprocess
from typing import Dict, List, Optional, Sequence
import functools

from e2e_test_artifacts import model_artifacts, iree_artifacts
from e2e_test_framework import serialization
from e2e_test_framework.definitions import iree_definitions

IREE_COMPILER_NAME = "iree-compile"


def _convert_to_cmd_string(cmds: Sequence[str]) -> str:
    if os.name == "nt":
        # list2cmdline is an undocumented method for Windows command lines. Python
        # doesn't provide an official method for quoting Windows command lines and
        # the correct implementation is slightly non-trivial. Use the undocumented
        # method for now and can be rewritten with our own implementation later.
        # See https://learn.microsoft.com/en-us/archive/blogs/twistylittlepassagesallalike/everyone-quotes-command-line-arguments-the-wrong-way
        return subprocess.list2cmdline(cmds)

    return " ".join(shlex.quote(cmd) for cmd in cmds)


def _dump_cmds_of_generation_config(
    gen_config: iree_definitions.ModuleGenerationConfig,
    root_path: pathlib.PurePath = pathlib.PurePath(),
):
    imported_model = gen_config.imported_model
    imported_model_path = iree_artifacts.get_imported_model_path(
        imported_model=imported_model, root_path=root_path
    )
    module_dir_path = iree_artifacts.get_module_dir_path(
        module_generation_config=gen_config, root_path=root_path
    )
    module_path = module_dir_path / iree_artifacts.MODULE_FILENAME
    compile_cmds = [
        IREE_COMPILER_NAME,
        str(imported_model_path),
        "-o",
        str(module_path),
    ]
    compile_cmds += gen_config.materialize_compile_flags(
        module_dir_path=module_dir_path
    )
    compile_cmd_str = _convert_to_cmd_string(compile_cmds)

    if imported_model.import_config.tool == iree_definitions.ImportTool.NONE:
        import_cmd_str = "# (Source model is already in MLIR)"
    else:
        source_model_path = model_artifacts.get_model_path(
            model=imported_model.model, root_path=root_path
        )
        import_cmds = [
            imported_model.import_config.tool.value,
            str(source_model_path),
            "-o",
            str(imported_model_path),
        ]
        import_cmds += imported_model.import_config.materialize_import_flags(
            model=imported_model.model
        )
        import_cmd_str = _convert_to_cmd_string(import_cmds)

    # Insert a blank line after each command to help read with line wrap.
    return ["Compile Module:", compile_cmd_str, "", "Import Model:", import_cmd_str, ""]


def _dump_cmds_from_run_config(
    run_config: iree_definitions.E2EModelRunConfig,
    root_path: pathlib.PurePath = pathlib.PurePath(),
):
    gen_config = run_config.module_generation_config
    module_path = (
        iree_artifacts.get_module_dir_path(
            module_generation_config=gen_config, root_path=root_path
        )
        / iree_artifacts.MODULE_FILENAME
    )

    run_cmds = [run_config.tool.value, f"--module={module_path}"]
    run_cmds += run_config.materialize_run_flags()
    # Insert a blank line after the command to help read with line wrap.
    lines = ["Run Module:", _convert_to_cmd_string(run_cmds), ""]
    lines += _dump_cmds_of_generation_config(gen_config=gen_config, root_path=root_path)
    return lines


def _dump_cmds_handler(
    e2e_test_artifacts_dir: pathlib.Path,
    execution_benchmark_config: Optional[pathlib.Path],
    compilation_benchmark_config: Optional[pathlib.Path],
    benchmark_id: Optional[str],
    **_unused_args,
):
    lines = []

    if execution_benchmark_config is not None:
        benchmark_groups = json.loads(execution_benchmark_config.read_text())
        for target_device, benchmark_group in benchmark_groups.items():
            shard_count = len(benchmark_group["shards"])
            for shard in benchmark_group["shards"]:
                run_configs = serialization.unpack_and_deserialize(
                    data=shard["run_configs"],
                    root_type=List[iree_definitions.E2EModelRunConfig],
                )
                for run_config in run_configs:
                    if (
                        benchmark_id is not None
                        and benchmark_id != run_config.composite_id
                    ):
                        continue

                    lines.append("################")
                    lines.append("")
                    lines.append(f"Execution Benchmark ID: {run_config.composite_id}")
                    lines.append(f"Name: {run_config}")
                    lines.append(f"Target Device: {target_device}")
                    lines.append(f"Shard: {shard['index']} / {shard_count}")
                    lines.append("")
                    lines += _dump_cmds_from_run_config(
                        run_config=run_config, root_path=e2e_test_artifacts_dir
                    )

    if compilation_benchmark_config is not None:
        benchmark_config = json.loads(compilation_benchmark_config.read_text())
        gen_configs = serialization.unpack_and_deserialize(
            data=benchmark_config["generation_configs"],
            root_type=List[iree_definitions.ModuleGenerationConfig],
        )
        for gen_config in gen_configs:
            if benchmark_id is not None and benchmark_id != gen_config.composite_id:
                continue

            lines.append("################")
            lines.append("")
            lines.append(f"Compilation Benchmark ID: {gen_config.composite_id}")
            lines.append(f"Name: {gen_config}")
            lines.append("")
            lines += _dump_cmds_of_generation_config(
                gen_config=gen_config, root_path=e2e_test_artifacts_dir
            )

    print(*lines, sep="\n")


# Represents a benchmark results file with the data already loaded from a JSON file.
class JSONBackedBenchmarkData:
    def __init__(self, source_filepath: pathlib.PurePath, data: Dict):
        if not isinstance(data, dict):
            raise ValueError(
                f"'{source_filepath}' seems not to be a valid benchmark-results-file (No JSON struct as root element)."
            )
        if "commit" not in data:
            raise ValueError(
                f"'{source_filepath}' seems not to be a valid benchmark-results-file ('commit' field not found)."
            )
        if "benchmarks" not in data:
            raise ValueError(
                f"'{source_filepath}' seems not to be a valid benchmark-results-file ('benchmarks' field not found)."
            )

        self.source_filepath: pathlib.PurePath = source_filepath
        self.data: Dict = data

    # Parses a JSON benchmark results file and makes some sanity checks
    @staticmethod
    def load_from_file(filepath: pathlib.Path):
        try:
            data = json.loads(filepath.read_bytes())
        except json.JSONDecodeError as e:
            raise ValueError(f"'{filepath}' seems not to be a valid JSON file: {e.msg}")
        return JSONBackedBenchmarkData(filepath, data)

    # A convenience wrapper around `loadFromFile` that accepts a sequence of paths and returns a sequence of JSONBackedBenchmarkData objects as a generator.
    @staticmethod
    def load_many_from_files(filepaths: Sequence[pathlib.Path]):
        return (
            JSONBackedBenchmarkData.load_from_file(filepath) for filepath in filepaths
        )


# Merges the benchmark results from `right` into `left` and returns the updated `left`
def _merge_two_resultsets(
    left: JSONBackedBenchmarkData, right: JSONBackedBenchmarkData
) -> JSONBackedBenchmarkData:
    if left.data["commit"] != right.data["commit"]:
        raise ValueError(
            f"'{right.source_filepath}' and the previous files are based on different commits ({left.data['commit']} != {right.data['commit']}). Merging not supported."
        )
    left.data["benchmarks"].extend(right.data["benchmarks"])
    return left


def merge_results(benchmark_results: Sequence[JSONBackedBenchmarkData]):
    return functools.reduce(_merge_two_resultsets, benchmark_results)


def _merge_results_handler(
    benchmark_results_files: Sequence[pathlib.Path], **_unused_args
):
    print(
        json.dumps(
            merge_results(
                JSONBackedBenchmarkData.load_many_from_files(benchmark_results_files)
            )
        )
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Miscellaneous tool to help work with benchmark suite and benchmark CI."
    )

    subparser = parser.add_subparsers(
        required=True, title="operation", dest="operation"
    )
    dump_cmds_parser = subparser.add_parser(
        "dump-cmds", help="Dump the commands to compile and run benchmarks manually."
    )
    dump_cmds_parser.add_argument(
        "--e2e_test_artifacts_dir",
        type=pathlib.PurePath,
        default=pathlib.Path(),
        help="E2E test artifacts root path used in the outputs of artifact paths",
    )
    dump_cmds_parser.add_argument(
        "--benchmark_id", type=str, help="Only dump the benchmark with this id"
    )
    dump_cmds_parser.add_argument(
        "--execution_benchmark_config",
        type=pathlib.Path,
        help="Config file exported from export_benchmark_config.py execution",
    )
    dump_cmds_parser.add_argument(
        "--compilation_benchmark_config",
        type=pathlib.Path,
        help="Config file exported from export_benchmark_config.py compilation",
    )
    dump_cmds_parser.set_defaults(handler=_dump_cmds_handler)

    merge_results_parser = subparser.add_parser(
        "merge-results",
        help="Merges the results from multiple benchmark results JSON files into a single JSON structure.",
    )
    merge_results_parser.add_argument(
        "benchmark_results_files",
        type=pathlib.Path,
        nargs="+",
        help="One or more benchmark results JSON file paths",
    )
    merge_results_parser.set_defaults(handler=_merge_results_handler)

    args = parser.parse_args()
    if (
        args.operation == "dump-cmds"
        and args.execution_benchmark_config is None
        and args.compilation_benchmark_config is None
    ):
        parser.error(
            "At least one of --execution_benchmark_config or "
            "--compilation_benchmark_config must be set."
        )

    return args


def main(args: argparse.Namespace):
    args.handler(**vars(args))


if __name__ == "__main__":
    main(_parse_arguments())
