from __future__ import annotations


import argparse
import contextlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


import rich_argparse


from bench_runner import benchmark_definitions
from bench_runner import flags as mflags
from bench_runner import git
from bench_runner.result import has_result
from bench_runner import runners
from bench_runner import util
from bench_runner.util import log_group, PathLike


from bench_runner.scripts import run_benchmarks as mrun_benchmarks


def get_windows_build_dir(force_32bit: bool) -> Path:
    if force_32bit:
        return Path("PCbuild") / "win32"
    return Path("PCbuild") / "amd64"


def get_exe_path(cpython: Path, flags: list[str], force_32bit: bool) -> Path:
    match util.get_simple_platform():
        case "linux":
            return cpython / "python"
        case "macos":
            return cpython / "python.exe"
        case "windows":
            build_dir = cpython / get_windows_build_dir(force_32bit)
            if "NOGIL" in flags:
                exe = next(build_dir.glob("python3.*.exe"))
            else:
                exe = build_dir / "python.exe"
            return exe


def run_in_venv(
    venv: PathLike, module: str, cmd: list[str], sudo: bool = False
) -> None:
    venv = Path(venv)

    if util.get_simple_platform() == "windows":
        exe = venv / "Scripts" / "python.exe"
    else:
        exe = venv / "bin" / "python"

    args = [
        str(exe),
        "-m",
        module,
        *cmd,
    ]

    if sudo:
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        args = ["sudo", f"LD_LIBRARY_PATH={ld_library_path}"] + args

    print("Running command:", " ".join(args))
    subprocess.check_call(args)


def should_run(
    force: bool,
    fork: str,
    ref: str,
    machine: str,
    pystats: bool,
    flags: list[str],
    cpython: Path = Path("cpython"),
    results_dir: Path = Path("results"),
) -> bool:
    try:
        commit_hash = git.get_git_hash(cpython)
    except subprocess.CalledProcessError:
        # This will fail if the cpython checkout failed for some reason. Print
        # a nice error message since the one the checkout itself gives is
        # totally inscrutable.
        print("The checkout of cpython failed.", file=sys.stderr)
        print(f"You specified fork {fork!r} and ref {ref!r}.", file=sys.stderr)
        print("Are you sure you entered the fork and ref correctly?", file=sys.stderr)
        # Fail the rest of the workflow
        sys.exit(1)

    found_result = has_result(
        results_dir,
        commit_hash,
        machine,
        pystats,
        flags,
        benchmark_definitions.get_benchmark_hash(),
        progress=False,
    )

    if force:
        if found_result is not None:
            for filepath in found_result.filename.parent.iterdir():
                if filepath.suffix != ".json":
                    git.remove(results_dir.parent, filepath)
        should_run = True
    else:
        should_run = (machine in ("__really_all", "all")) or found_result is None

    return should_run


def checkout_cpython(fork: str, ref: str, cpython: PathLike = Path("cpython")):
    git.clone(cpython, f"https://github.com/{fork}/cpython.git", branch=ref, depth=50)


def checkout_benchmarks():
    for repo in benchmark_definitions.BENCHMARK_REPOS:
        git.clone(
            Path(repo.dirname),
            repo.url,
            branch=repo.hash,
            depth=1,
        )


def compile_unix(
    cpython: PathLike,
    flags: list[str],
    pgo: bool,
    pystats: bool,
    reconfigure: bool = True,
) -> None:
    cpython = Path(cpython)
    runner = runners.get_runner_for_hostname()

    env = os.environ.copy()

    if util.get_simple_platform() == "macos":
        openssl_prefix = util.get_brew_prefix("openssl@1.1")
        env["PKG_CONFIG_PATH"] = f"{openssl_prefix}/lib/pkgconfig"

    args = []
    if pystats:
        args.append("--enable-pystats")
    if pgo:
        args.extend(["--enable-optimizations", "--with-lto=full"])
    if "PYTHON_UOPS" in flags:
        assert "JIT" not in flags
        args.append("--enable-experimental-jit=interpreter")
    if "JIT" in flags:
        assert "PYTHON_UOPS" not in flags
        args.append("--enable-experimental-jit=yes")
    if "NOGIL" in flags:
        args.append("--disable-gil")
    if "TAILCALL" in flags:
        args.append("--with-tail-call-interp")
    args.append("--enable-option-checking=fatal")
    if configure_flags := os.environ.get("PYTHON_CONFIGURE_FLAGS"):
        args.extend(shlex.split(configure_flags))

    make_args = []
    if (cores := runner.use_cores) is not None:
        make_args.extend(["-j", str(cores)])
    else:
        make_args.extend(["-j"])

    with contextlib.chdir(cpython):
        if reconfigure:
            subprocess.check_call(["./configure", *args], env=env)
        subprocess.check_call(["make", *make_args], env=env)


def compile_windows(
    cpython: PathLike, flags: list[str], pgo: bool, force_32bit: bool
) -> None:
    cpython = Path(cpython)

    args = ["--%"]  # This is the PowerShell "stop parsing" flag
    if force_32bit:
        args.extend(["-p", "win32"])
    args.extend(["-c", "Release"])
    if pgo:
        args.append("--pgo")
    if "JIT" in flags:
        args.append("--experimental-jit")
    if "PYTHON_UOPS" in flags:
        args.append("--experimental-jit-interpreter")
    if "NOGIL" in flags:
        args.append("--disable-gil")
    if "TAILCALL" in flags:
        args.append("--tail-call-interp")
    if configure_flags := os.environ.get("PYTHON_CONFIGURE_FLAGS"):
        args.append(configure_flags)

    with contextlib.chdir(cpython):
        subprocess.check_call(
            [
                "powershell.exe",
                Path("PCbuild") / "build.bat",
                *args,
            ],
        )
        shutil.copytree(get_windows_build_dir(force_32bit), "libs", dirs_exist_ok=True)


def clear_pip_cache(venv: PathLike) -> None:
    run_in_venv(venv, "pip", ["cache", "purge"])


def install_pyperformance(venv: PathLike) -> None:
    run_in_venv(venv, "pip", ["install", "./pyperformance"])


def tune_system(venv: PathLike, perf: bool) -> None:
    # System tuning is Linux only
    if util.get_simple_platform() != "linux":
        return

    args = ["system", perf and "reset" or "tune"]
    if cpu_affinity := os.environ.get("CPU_AFFINITY"):
        args.append(f'--affinity="{cpu_affinity}"')

    run_in_venv(venv, "pyperf", args, sudo=True)

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

    run_in_venv(
        venv,
        "pyperf",
        ["system", "reset"],
        sudo=True,
    )


def _main(
    fork: str,
    ref: str,
    machine: str,
    benchmarks: str,
    flags: list[str],
    force: bool,
    pgo: bool,
    perf: bool,
    pystats: bool,
    force_32bit: bool,
    run_id: str | None = None,
    fast: bool = False,
):
    venv = Path("venv")
    cpython = Path("cpython")
    platform = util.get_simple_platform()

    if force_32bit and platform != "windows":
        raise RuntimeError("32-bit builds are only supported on Windows")
    if perf and platform != "linux":
        raise RuntimeError("perf profiling is only supported on Linux")
    if pystats and platform != "linux":
        raise RuntimeError("Pystats is only supported on Linux")

    with log_group("Checking out CPython"):
        checkout_cpython(fork, ref, cpython)

    with log_group("Determining if we need to run benchmarks"):
        if not fast and not should_run(
            force, fork, ref, machine, False, flags, cpython=cpython
        ):
            print("No need to run benchmarks.  Skipping...")
            return

    with log_group("Checking out benchmarks"):
        checkout_benchmarks()

    with log_group("Compiling CPython"):
        match platform:
            case "linux" | "macos":
                compile_unix(cpython, flags, pgo, pystats)
            case "windows":
                compile_windows(cpython, flags, pgo, force_32bit)

        # Print out the version of Python we built just so we can confirm it's the
        # right thing in the logs
        subprocess.check_call([get_exe_path(cpython, flags, force_32bit), "-VV"])

    clear_pip_cache(venv)

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
                get_exe_path(cpython, flags, force_32bit),
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


def main():
    parser = argparse.ArgumentParser(
        description="""
        Run the full compile/benchmark workflow.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument("fork", help="The fork of CPython")
    parser.add_argument("ref", help="The git ref in the fork")
    parser.add_argument(
        "machine",
        help="The machine to run the benchmarks on.",
    )
    parser.add_argument("benchmarks", help="The benchmarks to run")
    parser.add_argument("flags", help="Configuration flags")
    parser.add_argument("--force", action="store_true", help="Force a re-run")
    parser.add_argument(
        "--pgo",
        action="store_true",
        help="Build with profiling guided optimization",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Collect Linux perf profiling data (Linux only)",
    )
    parser.add_argument(
        "--pystats",
        action="store_true",
        help="Enable Pystats (Linux only)",
    )
    parser.add_argument(
        "--32bit",
        action="store_true",
        dest="force_32bit",
        help="Do a 32-bit build (Windows only)",
    )
    parser.add_argument("--run_id", default=None, type=str, help="The github run id")
    parser.add_argument(
        "--_fast", action="store_true", help="Use fast mode, for testing"
    )
    args = parser.parse_args()

    _main(
        args.fork,
        args.ref,
        args.machine,
        args.benchmarks,
        mflags.parse_flags(args.flags),
        args.force,
        args.pgo,
        args.perf,
        args.pystats,
        args.force_32bit,
        args.run_id,
        args._fast,
    )


if __name__ == "__main__":
    main()
