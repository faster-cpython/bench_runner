import argparse
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import traceback


import numpy as np
import rich_argparse


from bench_runner import flags as mflags
from bench_runner import git
from bench_runner import result as mresult
from bench_runner.scripts import run_benchmarks
from bench_runner.scripts import workflow
from bench_runner.util import PathLike, format_seconds


def format_result(result: float | int, pystats: str | None = None) -> str:
    if pystats is None:
        return format_seconds(result)
    else:
        return f"{result:,}"


def _get_result_commandline(
    benchmark: str,
    good_val: float,
    bad_val: float,
    pgo: bool,
    flags: str,
    pystats: str | None,
    invert: bool,
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
        str(pystats),
        str(invert),
        str(repo.absolute()),
    ]


def parse_timing_result(benchmark_json: PathLike) -> float:
    # The name of the benchmark in the JSON file may be different from the one
    # used to select the benchmark. Therefore, just take the mean of all the
    # benchmarks in the JSON file.
    mresult.clear_contents_cache()
    r = mresult.Result.from_arbitrary_filename(benchmark_json)
    timing_data = r.get_timing_data()
    return float(np.mean([x.mean() for x in timing_data.values()]))


def get_result(
    benchmark: str,
    pgo: bool = False,
    flags: str = "",
    pystats: str | None = None,
    cpython: PathLike = Path("cpython"),
    reconfigure: bool = False,
) -> float:
    cpython = Path(cpython)
    python = cpython / "python"
    parsed_flags = mflags.parse_flags(flags)

    if pgo or reconfigure:
        # Jumping around through commits with PGO can leave stale PGO data
        # around, so we need to clean it each time.  We also always do it the
        # first time in case the *last* bisect run used pgo.
        subprocess.run(["make", "clean"], cwd=cpython)

    workflow.compile_unix(cpython, parsed_flags, pgo, pystats is not None, reconfigure)

    if pystats is None:
        run_benchmarks.run_benchmarks(python, benchmark)
        return parse_timing_result(run_benchmarks.BENCHMARK_JSON)
    else:
        run_benchmarks.collect_pystats(python, benchmark)
        summarize_stats_path = cpython / "Tools" / "scripts" / "summarize_stats.py"
        subprocess.check_output(
            [str(python), str(summarize_stats_path), "--json-output", "pystats.json"]
        )
        with open("pystats.json", "r") as f:
            contents = json.load(f)
            return int(contents[pystats])


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
    pystats: str | None = None,
    invert: bool = False,
    repo: PathLike = Path("."),
    cpython: PathLike = Path("cpython"),
):
    repo = Path(repo).absolute()
    cpython = Path(cpython).absolute()
    im = -1 if invert else 1

    delete_log()

    if not cpython.is_dir():
        git.clone(
            cpython, "https://github.com/python/cpython.git", branch="main", depth=None
        )

    git.checkout(cpython, good)
    good_result = get_result(
        benchmark, pgo, flags, pystats, cpython=cpython, reconfigure=True
    )
    log(f"KNOWN  GOOD ({good[:7]}): {format_result(good_result, pystats)}")

    git.checkout(cpython, bad)
    bad_result = get_result(benchmark, pgo, flags, pystats, cpython=cpython)
    log(f"KNOWN  BAD  ({bad[:7]}): {format_result(bad_result, pystats)}")

    if im * good_result >= im * bad_result:
        show_log()
        raise ValueError(
            f"Good result ({format_result(good_result, pystats)}) "
            f"must be {'more' if invert else 'less'} than "
            f"bad result ({format_result(bad_result, pystats)})."
        )

    try:
        with contextlib.chdir(cpython):
            subprocess.run(["git", "bisect", "start"])
            subprocess.run(["git", "bisect", "bad", bad])
            subprocess.run(["git", "bisect", "good", good])
            subprocess.run(
                ["git", "bisect", "run"]
                + _get_result_commandline(
                    benchmark,
                    good_result,
                    bad_result,
                    pgo,
                    flags,
                    pystats,
                    invert,
                    repo,
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
    parser.add_argument(
        "--pystats",
        type=str,
        help="Bisect using pystats. Should be the key in the pystats.json file.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the values, i.e. expect the bad value to be lower than "
        "the good value.",
    )

    args = parser.parse_args()

    _main(
        args.benchmark,
        args.good,
        args.bad,
        args.pgo,
        args.flags,
        args.pystats,
        args.invert,
    )


if __name__ == "__main__":
    # This is the entry point when we are called from `git bisect run` itself

    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=str)
    parser.add_argument("good_val", type=float)
    parser.add_argument("bad_val", type=float)
    parser.add_argument("pgo", type=str)
    parser.add_argument("flags", type=str)
    parser.add_argument("pystats", type=str)
    parser.add_argument("invert", type=str, choices=["True", "False"])
    parser.add_argument("repo", type=str)
    args = parser.parse_args()

    if args.pystats == "None":
        args.pystats = None

    invert = args.invert == "True"
    im = -1 if invert else 1

    mid_point = (args.good_val + args.bad_val) / 2.0

    repo = Path(args.repo)
    cpython = repo / "cpython"

    try:
        with contextlib.chdir(repo):
            result = get_result(
                args.benchmark,
                args.pgo == "True",
                args.flags,
                args.pystats,
                cpython=cpython,
            )
    except Exception as e:
        # If there was any exception, display that exception and traceback and
        # then abort the git bisect with -1
        traceback.print_exception(e)
        sys.exit(-1)

    # The confidence is 0.0 at the mid-point, 1.0 at the good and bad values,
    # and > 1.0 outside of that.
    confidence = abs((result - mid_point) / ((args.bad_val - args.good_val) / 2.0))

    with contextlib.chdir(repo):
        if im * result > im * mid_point:
            log(
                f"BISECT BAD  ({git.get_git_hash(cpython)[:7]}): "
                f"{format_result(result, args.pystats)} (confidence {confidence:.02f})"
            )
            print(f"BAD: {result} vs. ({args.good_val}, {args.bad_val})")
            sys.exit(1)
        else:
            log(
                f"BISECT GOOD ({git.get_git_hash(cpython)[:7]}): "
                f"{format_result(result, args.pystats)} (confidence {confidence:.02f})"
            )
            print(f"GOOD: {result} vs. ({args.good_val}, {args.bad_val})")
            sys.exit(0)
