from __future__ import annotations


import argparse
import csv
import json
import os
from operator import itemgetter
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Iterable


import rich_argparse


from bench_runner import flags
from bench_runner import git
from bench_runner.result import Result
from bench_runner.table import md_link
from bench_runner import util
from bench_runner.util import PathLike


REPO_ROOT = Path()
BENCHMARK_JSON = REPO_ROOT / "benchmark.json"
PROFILING_RESULTS = REPO_ROOT / "profiling" / "results"
GITHUB_URL = "https://github.com/" + os.environ.get(
    "GITHUB_REPOSITORY", "faster-cpython/bench_runner"
)
# Environment variables that control the execution of CPython
ENV_VARS = ["PYTHON_JIT"]


class NoBenchmarkError(Exception):
    pass


def get_benchmark_names(benchmarks: str) -> list[str]:
    benchmarks = "all" if benchmarks.strip() == "" else benchmarks

    output = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "pyperformance",
            "list",
            "--manifest",
            "benchmarks.manifest",
            "--benchmarks",
            benchmarks,
        ],
        encoding="utf-8",
    )

    # Process the output in a single list comprehension
    return [line[2:] for line in output.splitlines() if line.startswith("- ")]


def run_benchmarks(
    python: PathLike,
    benchmarks: str,
    command_prefix: Iterable[str] | None = None,
    test_mode: bool = False,
    extra_args: Iterable[str] | None = None,
) -> None:
    if benchmarks.strip() == "":
        benchmarks = "all"

    if BENCHMARK_JSON.is_file():
        BENCHMARK_JSON.unlink()

    if command_prefix is None:
        command_prefix = []

    if test_mode:
        fast_arg = ["--fast"]
    else:
        fast_arg = []

    if extra_args is None:
        extra_args = []

    args = [
        *command_prefix,
        sys.executable,
        "-m",
        "pyperformance",
        "run",
        *fast_arg,
        "-o",
        BENCHMARK_JSON,
        "--manifest",
        "benchmarks.manifest",
        "--benchmarks",
        benchmarks,
        "--python",
        python,
        "--inherit-environ",
        ",".join(ENV_VARS),
        *extra_args,
    ]

    print(f"RUNNING: {' '.join(str(x) for x in args)}")

    subprocess.call(args)

    # pyperformance frequently returns an error if any of the benchmarks failed.
    # We only want to fail if things are worse than that.

    if not BENCHMARK_JSON.is_file():
        raise NoBenchmarkError(
            f"No benchmark file created at {BENCHMARK_JSON.resolve()}."
        )
    with BENCHMARK_JSON.open() as fd:
        contents = json.load(fd)
    if len(contents.get("benchmarks", [])) == 0:
        raise NoBenchmarkError("No benchmarks were run.")


def collect_pystats(
    python: PathLike,
    benchmarks: str,
    fork: str,
    ref: str,
    individual: bool,
    flags: Iterable[str] | None = None,
) -> None:
    pystats_dir = Path("/tmp/py_stats")

    all_benchmarks = get_benchmark_names(benchmarks)

    extra_args = ["--same-loops", "loops.json", "--hook", "pystats"]

    if flags is None:
        flags = []

    # Clear all files in /tmp/py_stats. The _pystats.yml workflow already
    # does this, but this helps when running and testing things locally.
    for filename in pystats_dir.glob("*"):
        filename.unlink()

    # We could technically run each benchmark in parallel (since we don't care
    # about performance timings), however, since the stats are written to the
    # same directory, they would get intertwined. At some point, specifying an
    # output directory for stats might make sense for this.

    with tempfile.TemporaryDirectory() as tempdir:
        for benchmark in all_benchmarks:
            try:
                run_benchmarks(python, benchmark, extra_args=extra_args)
            except NoBenchmarkError:
                pass
            else:
                if individual:
                    run_summarize_stats(python, fork, ref, benchmark, flags=flags)

            for filename in pystats_dir.iterdir():
                os.rename(filename, Path(tempdir) / filename.name)

        for filename in Path(tempdir).iterdir():
            os.rename(filename, pystats_dir / filename.name)

        if individual:
            benchmark_links = all_benchmarks
        else:
            benchmark_links = []

        run_summarize_stats(python, fork, ref, "all", benchmark_links, flags=flags)


def perf_to_csv(lines: Iterable[str], output: PathLike):
    event_count_prefix = "# Event count (approx.): "
    total = None

    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith(event_count_prefix):
            total = int(line[len(event_count_prefix) :].strip())
        elif line.startswith("#") or line == "":
            pass
        elif total is None:
            raise ValueError("Could not find total sample count")
        else:
            _, period, command, _, symbol, shared, _ = line.split(maxsplit=6)
            pid, command = command.split(":")
            self_time = float(int(period)) / total
            if self_time > 0.0:
                rows.append([self_time, pid, command, shared, symbol])

    rows.sort(key=itemgetter(0), reverse=True)

    with Path(output).open("w") as fd:
        csvwriter = csv.writer(fd)
        csvwriter.writerow(["self", "pid", "command", "shared_obj", "symbol"])
        for row in rows:
            csvwriter.writerow(row)


def collect_perf(python: PathLike, benchmarks: str):
    all_benchmarks = get_benchmark_names(benchmarks)

    if PROFILING_RESULTS.is_dir():
        shutil.rmtree(PROFILING_RESULTS)
    PROFILING_RESULTS.mkdir()

    perf_data = Path("perf.data")
    for benchmark in all_benchmarks:
        if perf_data.exists():
            perf_data.unlink()

        try:
            run_benchmarks(
                python,
                benchmark,
                command_prefix=[
                    "perf",
                    "record",
                    "-o",
                    "perf.data",
                    "--",
                ],
            )
        except NoBenchmarkError:
            pass
        else:
            if perf_data.exists():
                output = subprocess.check_output(
                    [
                        "perf",
                        "report",
                        "--stdio",
                        "-g",
                        "none",
                        "--show-total-period",
                        "-s",
                        "pid,symbol,dso",
                        "-i",
                        "perf.data",
                    ],
                    encoding="utf-8",
                )
                perf_to_csv(
                    output.splitlines(), PROFILING_RESULTS / f"{benchmark}.perf.csv"
                )


def update_metadata(
    filename: PathLike,
    fork: str,
    ref: str,
    cpython: PathLike = Path("cpython"),
    run_id: str | None = None,
) -> None:
    with Path(filename).open() as fd:
        content = json.load(fd)

    metadata = content.setdefault("metadata", {})

    metadata["commit_id"] = git.get_git_hash(cpython)
    metadata["commit_fork"] = fork
    metadata["commit_branch"] = ref
    metadata["commit_date"] = git.get_git_commit_date(cpython)
    merge_base = git.get_git_merge_base(cpython)
    if merge_base is not None:
        metadata["commit_merge_base"] = merge_base
    metadata["benchmark_hash"] = util.get_benchmark_hash()
    if run_id is not None:
        metadata["github_action_url"] = f"{GITHUB_URL}/actions/runs/{run_id}"
    actor = os.environ.get("GITHUB_ACTOR")
    if actor is not None:
        metadata["github_actor"] = actor

    with Path(filename).open("w") as fd:
        json.dump(content, fd, indent=2)


def copy_to_directory(
    filename: PathLike, python: PathLike, fork: str, ref: str, flags: Iterable[str]
) -> None:
    result = Result.from_scratch(python, fork, ref, flags=flags)
    result.filename.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(filename, result.filename)


def run_summarize_stats(
    python: PathLike,
    fork: str,
    ref: str,
    benchmark: str,
    benchmarks: Iterable[str] | None = None,
    flags: Iterable[str] | None = None,
) -> None:
    if benchmarks is None:
        benchmarks = []

    if flags is None:
        flags = []

    summarize_stats_path = (
        Path(python).parent / "Tools" / "scripts" / "summarize_stats.py"
    )

    parts = ["pystats"]
    if benchmark != "all":
        parts.append(benchmark)
    result = Result.from_scratch(python, fork, ref, parts, flags=flags)
    result.filename.parent.mkdir(parents=True, exist_ok=True)
    pystats_json = result.filename.with_suffix(".json")

    args = [str(python), summarize_stats_path]
    args.extend(["--json-output", pystats_json])

    table = subprocess.check_output(args, encoding="utf-8")

    header = textwrap.dedent(
        f"""
    # Pystats results

    - benchmark: {benchmark}
    - fork: {fork}
    - ref: {ref}
    - commit hash: {git.get_git_hash(Path('cpython'))[:7]}
    - commit date: {git.get_git_commit_date(Path('cpython'))}

    """
    )

    with result.filename.with_suffix(".md").open("w") as fd:
        fd.write(header)
        if benchmarks:
            fd.write("- ")
            for name in benchmarks:
                fd.write(
                    md_link(
                        name,
                        str(Path(result.filename.name).with_suffix("")) + f"-{name}.md",
                    )
                )
                fd.write(", ")
            fd.write("\n")
        fd.write(table)

    if pystats_json.is_file():
        update_metadata(pystats_json, fork, ref)
    else:
        print(
            "WARNING: No pystats.json file generated. "
            "This is expected with CPython < 3.12"
        )


def select_benchmarks(benchmarks: str):
    if benchmarks == "all":
        return ",".join(
            ["all", *[f"-{x}" for x in util.get_excluded_benchmarks() if x]]
        )
    elif benchmarks == "all_and_excluded":
        return "all"
    return benchmarks


def _main(
    mode: str,
    python: PathLike,
    fork: str,
    ref: str,
    benchmarks: str,
    test_mode: bool,
    run_id: str | None,
    individual: bool,
    flags: Iterable[str],
) -> None:
    benchmarks = select_benchmarks(benchmarks)

    if mode == "benchmark":
        run_benchmarks(python, benchmarks, [], test_mode)
        update_metadata(BENCHMARK_JSON, fork, ref, run_id=run_id)
        copy_to_directory(BENCHMARK_JSON, python, fork, ref, flags)
    elif mode == "perf":
        collect_perf(python, benchmarks)
    elif mode == "pystats":
        collect_pystats(python, benchmarks, fork, ref, individual, flags)


def main():
    print("Environment variables:")
    for var in ENV_VARS:
        print(f"{var}={os.environ.get(var, '<unset>')}")

    parser = argparse.ArgumentParser(
        description="""
        Run benchmarks in `pyperformance` with the given python executable. Add
        additional metadata to a benchmark results file and then copy it to the
        correct location in the results tree.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "mode", choices=["benchmark", "perf", "pystats"], help="The mode of execution"
    )
    parser.add_argument("python", help="The path to the Python executable")
    parser.add_argument("fork", help="The fork of CPython")
    parser.add_argument("ref", help="The git ref in the fork")
    parser.add_argument("benchmarks", help="The benchmarks to run")
    parser.add_argument("flags", help="Configuration flags")
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run in a special mode for unit testing",
    )
    parser.add_argument("--run_id", default=None, type=str, help="The github run id")
    parser.add_argument(
        "--individual",
        action="store_true",
        help="For pystats mode, collect stats for each individual benchmark",
    )
    args = parser.parse_args()

    if args.test_mode:
        import socket

        def gethostname():
            return "pyperf"

        socket.gethostname = gethostname

        def dummy(*args, **kwargs):
            return None

        git.get_git_merge_base = dummy

    _main(
        args.mode,
        Path(args.python),
        args.fork,
        args.ref,
        args.benchmarks,
        args.test_mode,
        args.run_id,
        args.individual,
        flags.parse_flags(args.flags),
    )


if __name__ == "__main__":
    main()
