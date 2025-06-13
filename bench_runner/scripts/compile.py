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


def compile_unix(
    cpython: PathLike,
    flags: list[str],
    pgo: bool,
    pystats: bool,
    install_to: PathLike | None = None,
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
    if install_to is not None:
        install_to = Path(install_to)
        args.extend(["--prefix", str(install_to.resolve())])
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
        if install_to is not None:
            subprocess.check_call(["make", "install"], env=env)


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
        shutil.copytree(
            util.get_windows_build_dir(force_32bit), "libs", dirs_exist_ok=True
        )


def _main(
    fork: str,
    ref: str,
    machine: str,
    flags: list[str],
    force: bool,
    pgo: bool,
    pystats: bool,
    force_32bit: bool,
    install_to: PathLike | None = None,
):
    cpython = Path("cpython")
    platform = util.get_simple_platform()

    if force_32bit and platform != "windows":
        raise RuntimeError("32-bit builds are only supported on Windows")
    if pystats and platform != "linux":
        raise RuntimeError("Pystats is only supported on Linux")

    with log_group("Checking out CPython"):
        checkout_cpython(fork, ref, cpython)

    with log_group("Determining if we need to run benchmarks"):
        if not should_run(force, fork, ref, machine, False, flags, cpython=cpython):
            print("No need to run benchmarks.  Skipping...")
            return

    with log_group("Compiling CPython"):
        match platform:
            case "linux" | "macos":
                compile_unix(cpython, flags, pgo, pystats, install_to=install_to)
            case "windows":
                compile_windows(cpython, flags, pgo, force_32bit)

        # Print out the version of Python we built just so we can confirm it's the
        # right thing in the logs
        subprocess.check_call([util.get_exe_path(cpython, flags, force_32bit), "-VV"])


def add_compile_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("fork", help="The fork of CPython")
    parser.add_argument("ref", help="The git ref in the fork")
    parser.add_argument(
        "machine",
        help="The machine to run the benchmarks on.",
    )
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
    parser.add_argument(
        "--install-to",
        action="store",
        default=Path("cpython") / "install",
    )


def main():
    parser = argparse.ArgumentParser(
        description="""
        Compile a specific commit of CPython
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    add_compile_arguments(parser)
    args = parser.parse_args()

    _main(
        args.fork,
        args.ref,
        args.machine,
        mflags.parse_flags(args.flags),
        args.force,
        args.pgo,
        args.pystats,
        args.force_32bit,
        Path(args.install_to),
    )


if __name__ == "__main__":
    main()
