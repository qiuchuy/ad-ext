#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Runs all matched benchmark suites on an Android device.

This script probes the Android phone via `adb` and uses the device information
to filter and run suitable benchmarks and optionally captures Tracy traces on
the Android phone.

It expects that `adb` is installed, and there is iree tools cross-compiled
towards Android. If to capture traces, another set of tracing-enabled iree
tools and the Tracy `capture` tool should be cross-compiled towards Android.

Example usages:

  # Without trace generation
  python3 run_benchmarks.py \
    --normal_benchmark_tool_dir=/path/to/normal/android/target/tools/dir \
    /path/to/host/build/dir

  # With trace generation
  python3 run_benchmarks.py \
    --normal_benchmark_tool_dir=/path/to/normal/android/target/tools/dir \
    --traced_benchmark_tool_dir=/path/to/tracy/android/target/tools/dir \
    --trace_capture_tool=/path/to/host/build/tracy/capture \
    /path/to/host/build/dir
"""

import sys
import pathlib

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

import atexit
import json
import shutil
import subprocess
import tarfile
from typing import Any, Optional, Sequence, Tuple

from common import benchmark_suite as benchmark_suite_module
from common.benchmark_config import BenchmarkConfig
from common.benchmark_driver import BenchmarkDriver
from common.benchmark_definition import (
    DriverInfo,
    execute_cmd,
    execute_cmd_and_get_stdout,
    execute_cmd_and_get_output,
    get_git_commit_hash,
    get_iree_benchmark_module_arguments,
    wait_for_iree_benchmark_module_start,
    parse_iree_benchmark_metrics,
)
from common.benchmark_suite import MODEL_FLAGFILE_NAME, BenchmarkCase, BenchmarkSuite
from common.android_device_utils import (
    get_android_device_model,
    get_android_device_info,
    get_android_gpu_name,
)
import common.common_arguments
from e2e_test_artifacts import iree_artifacts
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.device_specs import device_parameters

# Root directory to perform benchmarks in on the Android device.
ANDROID_TMPDIR = pathlib.PurePosixPath("/data/local/tmp/iree-benchmarks")

NORMAL_TOOL_REL_DIR = pathlib.PurePosixPath("normal-tools")
TRACED_TOOL_REL_DIR = pathlib.PurePosixPath("traced-tools")


def adb_push_to_tmp_dir(
    content: pathlib.Path,
    relative_dir: pathlib.PurePosixPath = pathlib.PurePosixPath(),
    verbose: bool = False,
) -> pathlib.PurePosixPath:
    """Pushes content onto the Android device.

    Args:
      content: the full path to the source file.
      relative_dir: the directory to push to; relative to ANDROID_TMPDIR.

    Returns:
      The full path to the content on the Android device.
    """
    filename = content.name
    android_path = ANDROID_TMPDIR / relative_dir / filename
    # When the output is a TTY, keep the default progress info output.
    # In other cases, redirect progress info to null to avoid bloating log files.
    stdout_redirect = None if sys.stdout.isatty() else subprocess.DEVNULL
    execute_cmd(
        ["adb", "push", content.resolve(), android_path],
        verbose=verbose,
        stdout=stdout_redirect,
    )
    return android_path


def adb_execute_and_get_output(
    cmd_args: Sequence[str],
    relative_dir: pathlib.PurePosixPath = pathlib.PurePosixPath(),
    verbose: bool = False,
) -> Tuple[str, str]:
    """Executes command with adb shell.

    Switches to `relative_dir` relative to the android tmp directory before
    executing. Waits for completion and returns the command stdout.

    Args:
      cmd_args: a list containing the command to execute and its parameters
      relative_dir: the directory to execute the command in; relative to
        ANDROID_TMPDIR.

    Returns:
      Strings for stdout and stderr.
    """
    cmd = ["adb", "shell", "cd", ANDROID_TMPDIR / relative_dir, "&&"]
    cmd.extend(cmd_args)
    return execute_cmd_and_get_output(cmd, verbose=verbose)


def adb_execute(
    cmd_args: Sequence[str],
    relative_dir: pathlib.PurePosixPath = pathlib.PurePosixPath(),
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    """Executes command with adb shell.

    Switches to `relative_dir` relative to the android tmp directory before
    executing. Waits for completion. Output is streamed to the terminal.

    Args:
      cmd_args: a list containing the command to execute and its parameters
      relative_dir: the directory to execute the command in; relative to
        ANDROID_TMPDIR.

    Returns:
      The completed process.
    """
    cmd = ["adb", "shell", "cd", ANDROID_TMPDIR / relative_dir, "&&"]
    cmd.extend(cmd_args)
    return execute_cmd(cmd, verbose=verbose)


def is_magisk_su():
    """Returns true if the Android device has a Magisk SU binary."""
    stdout, _ = adb_execute_and_get_output(["su", "--help"])
    return "MagiskSU" in stdout


def adb_execute_as_root(cmd_args: Sequence[Any]) -> subprocess.CompletedProcess:
    """Executes the given command as root."""
    cmd = ["su", "-c" if is_magisk_su() else "root"]
    cmd.extend(cmd_args)
    return adb_execute(cmd)


def adb_start_cmd(
    cmd_args: Sequence[str],
    relative_dir: pathlib.PurePosixPath = pathlib.PurePosixPath(),
    verbose: bool = False,
) -> subprocess.Popen:
    """Executes command with adb shell in a directory and returns the handle
    without waiting for completion.

    Args:
      cmd_args: a list containing the command to execute and its parameters
      relative_dir: the directory to execute the command in; relative to
        ANDROID_TMPDIR.

    Returns:
      A Popen object for the started command.
    """
    cmd = ["adb", "shell", "cd", ANDROID_TMPDIR / relative_dir, "&&"]
    cmd.extend(cmd_args)

    if verbose:
        print(f"cmd: {cmd}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)


def get_vmfb_full_path_for_benchmark_case(
    benchmark_case_dir: pathlib.Path,
) -> pathlib.Path:
    flagfile = benchmark_case_dir / MODEL_FLAGFILE_NAME
    for line in flagfile.read_text().splitlines():
        flag_name, flag_value = line.strip().split("=")
        if flag_name == "--module":
            # Realpath canonicalization matters. The caller may rely on that to track
            # which files it already pushed.
            return (benchmark_case_dir / flag_value).resolve()
    raise ValueError(f"{flagfile} does not contain a --module flag")


class AndroidBenchmarkDriver(BenchmarkDriver):
    """Android benchmark driver."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.already_pushed_files = {}

    def run_benchmark_case(
        self,
        benchmark_case: BenchmarkCase,
        benchmark_results_filename: Optional[pathlib.Path],
        capture_filename: Optional[pathlib.Path],
    ) -> None:
        benchmark_case_dir = benchmark_case.benchmark_case_dir
        android_case_dir = pathlib.PurePosixPath(
            benchmark_case_dir.relative_to(self.config.root_benchmark_dir)
        )

        self.__check_and_push_file(
            benchmark_case_dir / iree_artifacts.MODULE_FILENAME, android_case_dir
        )
        run_config = benchmark_case.run_config
        taskset = self.__deduce_taskset_from_run_config(run_config)
        run_args = run_config.materialize_run_flags()
        run_args.append(f"--module={iree_artifacts.MODULE_FILENAME}")

        if benchmark_results_filename is not None:
            self.__run_benchmark(
                android_case_dir=android_case_dir,
                benchmark_case=benchmark_case,
                run_args=run_args,
                results_filename=benchmark_results_filename,
                taskset=taskset,
            )

        if capture_filename is not None:
            self.__run_capture(
                android_case_dir=android_case_dir,
                benchmark_case=benchmark_case,
                run_args=run_args,
                capture_filename=capture_filename,
                taskset=taskset,
            )

    def __run_benchmark(
        self,
        android_case_dir: pathlib.PurePosixPath,
        benchmark_case: BenchmarkCase,
        run_args: Sequence[str],
        results_filename: pathlib.Path,
        taskset: str,
    ):
        if self.config.normal_benchmark_tool_dir is None:
            raise ValueError("normal_benchmark_tool_dir can't be None.")

        tool_name = benchmark_case.benchmark_tool_name
        host_tool_path = self.config.normal_benchmark_tool_dir / tool_name
        android_tool = self.__check_and_push_file(host_tool_path, NORMAL_TOOL_REL_DIR)
        cmd = ["taskset", taskset, android_tool]
        cmd += run_args
        if tool_name == "iree-benchmark-module":
            cmd += get_iree_benchmark_module_arguments(
                results_filename=f"'{results_filename.name}'",
                driver_info=benchmark_case.driver_info,
                benchmark_min_time=self.config.benchmark_min_time,
            )

        benchmark_stdout, benchmark_stderr = adb_execute_and_get_output(
            cmd, android_case_dir, verbose=self.verbose
        )
        benchmark_metrics = parse_iree_benchmark_metrics(
            benchmark_stdout, benchmark_stderr
        )
        if self.verbose:
            print(benchmark_metrics)
        results_filename.write_text(json.dumps(benchmark_metrics.to_json_object()))

    def __run_capture(
        self,
        android_case_dir: pathlib.PurePosixPath,
        benchmark_case: BenchmarkCase,
        run_args: Sequence[str],
        capture_filename: pathlib.Path,
        taskset: str,
    ):
        capture_config = self.config.trace_capture_config
        if capture_config is None:
            raise ValueError("capture_config can't be None.")

        tool_name = benchmark_case.benchmark_tool_name
        host_tool_path = capture_config.traced_benchmark_tool_dir / tool_name
        android_tool = self.__check_and_push_file(host_tool_path, TRACED_TOOL_REL_DIR)
        run_cmd = [
            "TRACY_NO_EXIT=1",
            f"IREE_PRESERVE_DYLIB_TEMP_FILES={ANDROID_TMPDIR}",
            "taskset",
            taskset,
            android_tool,
        ]
        run_cmd += run_args
        if tool_name == "iree-benchmark-module":
            run_cmd += get_iree_benchmark_module_arguments(
                driver_info=benchmark_case.driver_info,
                benchmark_min_time=self.config.benchmark_min_time,
                capture_mode=True,
            )

        # Just launch the traced benchmark tool with TRACY_NO_EXIT=1 without
        # waiting for the adb command to complete as that won't happen.
        process = adb_start_cmd(run_cmd, android_case_dir, verbose=self.verbose)

        wait_for_iree_benchmark_module_start(process, self.verbose)

        # Now it's okay to collect the trace via the capture tool. This will
        # send the signal to let the previously waiting benchmark tool to
        # complete.
        capture_cmd = [capture_config.trace_capture_tool, "-f", "-o", capture_filename]
        # If verbose, just let the subprocess print its output. The subprocess
        # may need to detect if the output is a TTY to decide whether to log
        # verbose progress info and use ANSI colors, so it's better to use
        # stdout redirection than to capture the output in a string.
        stdout_redirect = None if self.verbose else subprocess.DEVNULL
        execute_cmd(capture_cmd, verbose=self.verbose, stdout=stdout_redirect)

    # TODO(#13187): These logics are inherited from the legacy benchmark suites,
    # which only work for a few specific phones. We should define the topology
    # in their device specs.
    def __deduce_taskset_from_run_config(
        self, run_config: iree_definitions.E2EModelRunConfig
    ) -> str:
        """Deduces the CPU mask according to device and execution config."""

        device_spec = run_config.target_device_spec
        # For GPU benchmarks, use the most performant core.
        if device_spec.architecture.type == common_definitions.ArchitectureType.GPU:
            return "80"

        device_params = device_spec.device_parameters
        single_thread = "1-thread" in run_config.module_execution_config.tags
        if device_parameters.ARM_BIG_CORES in device_params:
            return "80" if single_thread else "f0"
        elif device_parameters.ARM_LITTLE_CORES in device_params:
            return "08" if single_thread else "0f"
        elif device_parameters.ALL_CORES in device_params:
            return "80" if single_thread else "ff"

        raise ValueError(f"Unsupported config to deduce taskset: '{run_config}'.")

    def __check_and_push_file(
        self, host_path: pathlib.Path, relative_dir: pathlib.PurePosixPath
    ):
        """Checks if the file has been pushed and pushes it if not."""
        android_path = self.already_pushed_files.get(host_path)
        if android_path is not None:
            return android_path

        android_path = adb_push_to_tmp_dir(
            host_path, relative_dir=relative_dir, verbose=self.verbose
        )
        self.already_pushed_files[host_path] = android_path
        return android_path


def set_cpu_frequency_scaling_governor(governor: str):
    git_root = execute_cmd_and_get_stdout(["git", "rev-parse", "--show-toplevel"])
    cpu_script = (
        pathlib.Path(git_root)
        / "build_tools"
        / "benchmarks"
        / "set_android_scaling_governor.sh"
    )
    android_path = adb_push_to_tmp_dir(cpu_script)
    adb_execute_as_root([android_path, governor])


def set_gpu_frequency_scaling_policy(policy: str):
    git_root = execute_cmd_and_get_stdout(["git", "rev-parse", "--show-toplevel"])
    device_model = get_android_device_model()
    gpu_name = get_android_gpu_name()
    benchmarks_tool_dir = pathlib.Path(git_root) / "build_tools" / "benchmarks"
    if device_model == "Pixel-6" or device_model == "Pixel-6-Pro":
        gpu_script = benchmarks_tool_dir / "set_pixel6_gpu_scaling_policy.sh"
    elif gpu_name.lower().startswith("adreno"):
        gpu_script = benchmarks_tool_dir / "set_adreno_gpu_scaling_policy.sh"
    else:
        raise RuntimeError(
            f"Unsupported device '{device_model}' for setting GPU scaling policy"
        )
    android_path = adb_push_to_tmp_dir(gpu_script)
    adb_execute_as_root([android_path, policy])


def main(args):
    device_info = get_android_device_info(args.verbose)
    if args.verbose:
        print(device_info)

    commit = get_git_commit_hash("HEAD")
    benchmark_config = BenchmarkConfig.build_from_args(args, commit)
    benchmark_groups = json.loads(args.execution_benchmark_config.read_text())
    run_configs = benchmark_suite_module.get_run_configs_by_target_and_shard(
        benchmark_groups, args.target_device_name, args.shard_index
    )

    benchmark_suite = BenchmarkSuite.load_from_run_configs(
        run_configs=run_configs, root_benchmark_dir=benchmark_config.root_benchmark_dir
    )

    benchmark_driver = AndroidBenchmarkDriver(
        device_info=device_info,
        benchmark_config=benchmark_config,
        benchmark_suite=benchmark_suite,
        benchmark_grace_time=1.0,
        verbose=args.verbose,
    )

    if args.pin_cpu_freq:
        set_cpu_frequency_scaling_governor("performance")
        atexit.register(set_cpu_frequency_scaling_governor, "schedutil")
    if args.pin_gpu_freq:
        set_gpu_frequency_scaling_policy("performance")
        atexit.register(set_gpu_frequency_scaling_policy, "default")

    # Clear the benchmark directory on the Android device first just in case
    # there are leftovers from manual or failed runs.
    execute_cmd_and_get_stdout(
        ["adb", "shell", "rm", "-rf", ANDROID_TMPDIR], verbose=args.verbose
    )

    if not args.no_clean:
        # Clear the benchmark directory on the Android device.
        atexit.register(
            execute_cmd_and_get_stdout,
            ["adb", "shell", "rm", "-rf", ANDROID_TMPDIR],
            verbose=args.verbose,
        )
        # Also clear temporary directory on the host device.
        atexit.register(shutil.rmtree, args.tmp_dir)

    # Tracy client and server communicate over port 8086 by default. If we want
    # to capture traces along the way, forward port via adb.
    trace_capture_config = benchmark_config.trace_capture_config
    if trace_capture_config:
        execute_cmd_and_get_stdout(
            ["adb", "forward", "tcp:8086", "tcp:8086"], verbose=args.verbose
        )
        atexit.register(
            execute_cmd_and_get_stdout,
            ["adb", "forward", "--remove", "tcp:8086"],
            verbose=args.verbose,
        )

    benchmark_driver.run()

    benchmark_results = benchmark_driver.get_benchmark_results()
    if args.output is not None:
        with open(args.output, "w") as f:
            f.write(benchmark_results.to_json_str())

    if args.verbose:
        print(benchmark_results.commit)
        print(benchmark_results.benchmarks)

    if trace_capture_config:
        # Put all captures in a tarball and remove the original files.
        with tarfile.open(trace_capture_config.capture_tarball, "w:gz") as tar:
            for capture_filename in benchmark_driver.get_capture_filenames():
                tar.add(capture_filename)

    benchmark_errors = benchmark_driver.get_benchmark_errors()
    if benchmark_errors:
        print("Benchmarking completed with errors", file=sys.stderr)
        raise RuntimeError(benchmark_errors)


if __name__ == "__main__":
    main(common.common_arguments.Parser().parse_args())
