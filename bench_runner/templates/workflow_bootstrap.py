# This script may only use the standard library, since it bootstraps setting up
# the virtual environment to run the full bench_runner.


# NOTE: This file should import in Python 3.9 or later so it can at least print
# the error message that the version of Python is too old.


import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def create_venv(venv: Path) -> None:
    if venv.exists():
        shutil.rmtree(venv)

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "venv",
            str(venv),
        ]
    )


def run_in_venv(
    venv: Path, module: str, cmd: list[str], prefix: list[str] = []
) -> None:
    venv = Path(venv)

    if sys.platform == "win32":
        exe = Path("Scripts") / "python.exe"
    else:
        exe = Path("bin") / "python"

    args = [
        *prefix,
        str(venv / exe),
        "-m",
        module,
        *cmd,
    ]

    print("Running command:", " ".join(args))
    subprocess.check_call(args)


def install_requirements(venv: Path) -> None:
    run_in_venv(venv, "pip", ["install", "--upgrade", "pip"])
    run_in_venv(venv, "pip", ["install", "-r", "requirements.txt"])


def _main(
    fork: str,
    ref: str,
    machine: str,
    benchmarks: str,
    flags: str,
    force: bool,
    pgo: bool,
    perf: bool,
    pystats: bool,
    force_32bit: bool,
    run_id: str | None = None,
):
    if force_32bit and sys.platform != "win32":
        raise RuntimeError("32-bit builds are only supported on Windows")
    if perf and not sys.platform.startswith("linux"):
        raise RuntimeError("perf profiling is only supported on Linux")
    if pystats and not sys.platform.startswith("linux"):
        raise RuntimeError("Pystats is only supported on Linux")

    venv = Path("venv")
    create_venv(venv)
    install_requirements(venv)

    # Now that we've installed the full bench_runner library,
    # continue on in a new process...

    args = ["workflow", fork, ref, machine, benchmarks, flags]
    if force:
        args.append("--force")
    if pgo:
        args.append("--pgo")
    if perf:
        args.append("--perf")
    if pystats:
        args.append("--pystats")
    if force_32bit:
        args.append("--32bit")
    if run_id:
        args.extend(["--run_id", run_id])

    run_in_venv(venv, "bench_runner", args)


def main():
    parser = argparse.ArgumentParser(
        description="""
        Run the full compile/benchmark workflow.
        """,
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
    args = parser.parse_args()

    _main(
        args.fork,
        args.ref,
        args.machine,
        args.benchmarks,
        args.flags,
        args.force,
        args.pgo,
        args.perf,
        args.pystats,
        args.force_32bit,
        args.run_id,
    )


if __name__ == "__main__":
    if sys.version_info[:2] < (3, 11):
        print(
            "The benchmarking infrastructure requires Python 3.11 or later.",
            file=sys.stderr,
        )
        sys.exit(1)

    main()
