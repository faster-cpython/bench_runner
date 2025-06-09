import argparse
import contextlib
from pathlib import Path
import subprocess
import sys
import traceback


import numpy as np
import rich_argparse


from bench_runner import flags as mflags
from bench_runner import git
from bench_runner import result
from bench_runner.scripts import run_benchmarks
from bench_runner.scripts import workflow
from bench_runner.util import PathLike, format_seconds


def _get_result_commandline(
    benchmark: str,
    good_val: float,
    bad_val: float,
    pgo: bool,
    flags: str,
    repo: PathLike,
) -> list[str]:
    repo = Path(repo)

    return [
        sys.executable,
        "-m",
        "bench_runner.scripts.bisect",
        benchmark,
        str(good_val),
        str(bad_val),
        str(pgo),
        str(flags),
        str(repo.absolute()),
    ]


def parse_result(benchmark_json: PathLike) -> float:
    # The name of the benchmark in the JSON file may be different from the one
    # used to select the benchmark. Therefore, just take the mean of all the
    # benchmarks in the JSON file.
    result.clear_contents_cache()
    r = result.Result.from_arbitrary_filename(benchmark_json)
    timing_data = r.get_timing_data()
    return float(np.mean([x.mean() for x in timing_data.values()]))


def get_result(
    benchmark: str,
    pgo: bool = False,
    flags: str = "",
    cpython: PathLike = Path("cpython"),
    reconfigure: bool = False,
) -> float:
    cpython = Path(cpython)

    if pgo or reconfigure:
        # Jumping around through commits with PGO can leave stale PGO data
        # around, so we need to clean it each time.  We also always do it the
        # first time in case the *last* bisect run used pgo.
        subprocess.run(["make", "clean"], cwd=cpython)

    workflow.compile_unix(cpython, mflags.parse_flags(flags), pgo, False, reconfigure)
    run_benchmarks.run_benchmarks(cpython / "python", benchmark)
    timing = parse_result(run_benchmarks.BENCHMARK_JSON)

    return timing


def get_log_file() -> Path:
    return Path("bisect_log.txt")


def delete_log() -> None:
    bisect_log = get_log_file()
    if bisect_log.is_file():
        bisect_log.unlink()


def show_log() -> None:
    print()
    print("Bisect log:")

    with get_log_file().open("r") as f:
        for line in f.readlines():
            print(line.strip())


def log(message: str) -> None:
    with get_log_file().open("a") as f:
        f.write(f"{message}\n")


def _main(
    benchmark: str,
    good: str,
    bad: str,
    pgo: bool = False,
    flags: str = "",
    repo: PathLike = Path("."),
    cpython: PathLike = Path("cpython"),
):
    repo = Path(repo).absolute()
    cpython = Path(cpython).absolute()

    delete_log()

    if not cpython.is_dir():
        git.clone(
            cpython, "https://github.com/python/cpython.git", branch="main", depth=None
        )

    git.checkout(cpython, good)
    good_timing = get_result(benchmark, pgo, flags, cpython=cpython, reconfigure=True)
    log(f"KNOWN  GOOD ({good[:7]}): {format_seconds(good_timing)}")

    git.checkout(cpython, bad)
    bad_timing = get_result(benchmark, pgo, flags, cpython=cpython)
    log(f"KNOWN  BAD  ({bad[:7]}): {format_seconds(bad_timing)}")

    if good_timing >= bad_timing:
        show_log()
        raise ValueError(
            f"Good timing ({good_timing}) must be less than bad timing ({bad_timing})."
        )

    try:
        with contextlib.chdir(cpython):
            subprocess.run(["git", "bisect", "start"])
            subprocess.run(["git", "bisect", "bad", bad])
            subprocess.run(["git", "bisect", "good", good])
            subprocess.run(
                ["git", "bisect", "run"]
                + _get_result_commandline(
                    benchmark, good_timing, bad_timing, pgo, flags, repo
                )
            )
    finally:
        show_log()
        delete_log()


def main():
    # This is the entry point for the user

    parser = argparse.ArgumentParser(
        description="""
        Run bisect on a benchmark to find the first regressing commit.

        A full checkout of CPython should be in the cpython directory.
        If it doesn't exist, it will be cloned.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "benchmark",
        type=str,
        help="The benchmark to run bisect on.",
    )
    parser.add_argument(
        "good",
        type=str,
        help="The good commit hash for the bisect.",
    )
    parser.add_argument(
        "bad",
        type=str,
        help="The bad commit hash for the bisect.",
    )
    parser.add_argument(
        "--pgo",
        action="store_true",
    )
    parser.add_argument(
        "--flags",
        type=str,
        default="",
    )

    args = parser.parse_args()

    _main(args.benchmark, args.good, args.bad, args.pgo, args.flags)


if __name__ == "__main__":
    # This is the entry point when we are called from `git bisect run` itself

    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=str)
    parser.add_argument("good_val", type=float)
    parser.add_argument("bad_val", type=float)
    parser.add_argument("pgo", type=str)
    parser.add_argument("flags", type=str)
    parser.add_argument("repo", type=str)
    args = parser.parse_args()

    mid_point = (args.good_val + args.bad_val) / 2.0

    repo = Path(args.repo)
    cpython = repo / "cpython"

    try:
        with contextlib.chdir(repo):
            timing = get_result(
                args.benchmark, args.pgo == "True", args.flags, cpython=cpython
            )
    except Exception as e:
        # If there was any exception, display that exception and traceback and
        # then abort the git bisect with -1
        traceback.print_exception(e)
        sys.exit(-1)

    # The confidence is 0.0 at the mid-point, 1.0 at the good and bad values,
    # and > 1.0 outside of that.
    confidence = abs((timing - mid_point) / ((args.bad_val - args.good_val) / 2.0))

    with contextlib.chdir(repo):
        if timing > mid_point:
            log(
                f"BISECT BAD  ({git.get_git_hash(cpython)[:7]}): "
                f"{format_seconds(timing)} (confidence {confidence:.02f})"
            )
            print(f"BAD: {timing} vs. ({args.good_val}, {args.bad_val})")
            sys.exit(1)
        else:
            log(
                f"BISECT GOOD ({git.get_git_hash(cpython)[:7]}): "
                f"{format_seconds(timing)} (confidence {confidence:.02f})"
            )
            print(f"GOOD: {timing} vs. ({args.good_val}, {args.bad_val})")
            sys.exit(0)
