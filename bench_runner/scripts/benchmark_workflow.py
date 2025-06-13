from __future__ import annotations


import argparse
import os
from pathlib import Path
import shutil
import subprocess


import rich_argparse


from bench_runner import benchmark_definitions
from bench_runner import flags as mflags
from bench_runner import git
from bench_runner import util
from bench_runner.util import log_group, PathLike


from bench_runner.scripts import compile as mcompile
from bench_runner.scripts import run_benchmarks as mrun_benchmarks


def checkout_benchmarks():
    for repo in benchmark_definitions.BENCHMARK_REPOS:
        git.clone(
            Path(repo.dirname),
            repo.url,
            branch=repo.hash,
            depth=1,
        )


def install_pyperformance(venv: PathLike) -> None:
    util.run_in_venv(venv, "pip", ["install", "./pyperformance"])


def tune_system(venv: PathLike, perf: bool) -> None:
    # System tuning is Linux only
    if util.get_simple_platform() != "linux":
        return

    args = ["system", perf and "reset" or "tune"]
    if cpu_affinity := os.environ.get("CPU_AFFINITY"):
        args.append(f'--affinity="{cpu_affinity}"')

    util.run_in_venv(venv, "pyperf", args, sudo=True)

    if perf:
        subprocess.check_call(
            [
                "sudo",
                "bash",
                "-c",
                "echo 100000 > /proc/sys/kernel/perf_event_max_sample_rate",
            ]
        )


def reset_system(venv: PathLike) -> None:
    # System tuning is Linux only
    if util.get_simple_platform() != "linux":
        return

    util.run_in_venv(
        venv,
        "pyperf",
        ["system", "reset"],
        sudo=True,
    )


def _main(
    fork: str,
    ref: str,
    benchmarks: str,
    flags: list[str],
    perf: bool,
    pystats: bool,
    force_32bit: bool,
    run_id: str | None = None,
    fast: bool = False,
):
    venv = Path("venv")
    cpython = Path("install")
    platform = util.get_simple_platform()

    if force_32bit and platform != "windows":
        raise RuntimeError("32-bit builds are only supported on Windows")
    if perf and platform != "linux":
        raise RuntimeError("perf profiling is only supported on Linux")
    if pystats and platform != "linux":
        raise RuntimeError("Pystats is only supported on Linux")

    with log_group("Checking out benchmarks"):
        checkout_benchmarks()

    with log_group("Installing pyperformance"):
        install_pyperformance(venv)

    if not fast:
        with log_group("Tuning system"):
            tune_system(venv, perf)

    try:
        if Path(".debug").exists():
            shutil.rmtree(".debug")

        pystats_dir = Path("/tmp") / "py_stats"
        if pystats:
            shutil.rmtree(pystats_dir, ignore_errors=True)
            pystats_dir.mkdir(parents=True)

        if perf:
            mode = "perf"
        elif pystats:
            mode = "pystats"
        else:
            mode = "benchmark"

        with log_group("Running benchmarks"):
            mrun_benchmarks._main(
                mode,
                util.get_exe_path(cpython, flags, force_32bit),
                fork,
                ref,
                benchmarks,
                flags=flags,
                run_id=run_id,
                test_mode=fast,
                individual=pystats,
            )
    finally:
        if not fast:
            reset_system(venv)


def add_benchmark_workflow_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("benchmarks", help="The benchmarks to run")
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Collect Linux perf profiling data (Linux only)",
    )
    parser.add_argument("--run_id", default=None, type=str, help="The github run id")
    parser.add_argument(
        "--_fast", action="store_true", help="Use fast mode, for testing"
    )


def main():
    parser = argparse.ArgumentParser(
        description="""
        Run the full compile/benchmark workflow.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    mcompile.add_compile_arguments(parser)
    add_benchmark_workflow_arguments(parser)
    args = parser.parse_args()

    _main(
        args.fork,
        args.ref,
        args.benchmarks,
        mflags.parse_flags(args.flags),
        args.perf,
        args.pystats,
        args.force_32bit,
        args.run_id,
        args._fast,
    )


if __name__ == "__main__":
    main()
