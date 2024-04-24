from __future__ import annotations


import argparse
import csv
import multiprocessing
import os
from operator import itemgetter
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Iterable


import ujson


from bench_runner import git
from bench_runner.result import Result
from bench_runner.table import md_link
from bench_runner import util


REPO_ROOT = Path()
PROFILING_RESULTS = REPO_ROOT / "profiling" / "results"
GITHUB_URL = "https://github.com/" + os.environ.get(
    "GITHUB_REPOSITORY", "faster-cpython/bench_runner"
)
# Environment variables that control the execution of CPython
ENV_VARS = ["PYTHON_UOPS", "PYTHON_PYSTATS_DIR"]


class NoBenchmarkError(Exception):
    pass


def get_benchmark_names(benchmarks: str) -> list[str]:
    if benchmarks.strip() == "":
        benchmarks = "all"

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

    return [line[2:].strip() for line in output.splitlines() if line.startswith("- ")]


def run_benchmarks(
    python: Path | str,
    benchmarks: str,
    command_prefix: Iterable[str] | None = None,
    test_mode: bool = False,
    extra_args: Iterable[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    if cwd is None:
        cwd = Path()

    benchmark_json = cwd / "benchmark.json"

    if benchmarks.strip() == "":
        benchmarks = "all"

    if benchmark_json.is_file():
        benchmark_json.unlink()

    if command_prefix is None:
        command_prefix = []

    if test_mode:
        fast_arg = ["--fast"]
    else:
        fast_arg = []

    if extra_args is None:
        extra_args = []

    if env is None:
        env = {}

    args = [
        *command_prefix,
        sys.executable,
        "-m",
        "pyperformance",
        "run",
        *fast_arg,
        "-o",
        benchmark_json.resolve(),
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

    print(f"RUNNING: {args}")

    subprocess.call(args, env=env, cwd=cwd)

    # pyperformance frequently returns an error if any of the benchmarks failed.
    # We only want to fail if things are worse than that.

    if not benchmark_json.is_file():
        raise NoBenchmarkError(
            f"No benchmark file created at {benchmark_json.resolve()}."
        )
    with open(benchmark_json) as fd:
        contents = ujson.load(fd)
    if len(contents.get("benchmarks", [])) == 0:
        raise NoBenchmarkError("No benchmarks were run.")


def _collect_single_pystats(
    cumulative_stats_dir: Path,
    python: Path,
    benchmark: str,
    fork: str,
    ref: str,
    flags: Iterable[str],
    individual: bool,
):
    extra_args = ["--same-loops", "loops.json"]

    pid = os.getpid()
    root_dir = Path("..")
    working_dir = Path(f"pystats{pid:08}")
    if working_dir.exists():
        raise RuntimeError(f"Working directory {working_dir} already exists")
    working_dir.mkdir()

    try:
        # Set up the working directory with the necessary content and symlinks
        shutil.copyfile("benchmarks.manifest", working_dir / "benchmarks.manifest")
        shutil.copyfile("loops.json", working_dir / "loops.json")
        (working_dir / "cpython").symlink_to(root_dir / "cpython")
        (working_dir / "pyperformance").symlink_to(root_dir / "pyperformance")
        (working_dir / "pyston-benchmarks").symlink_to(root_dir / "pyston-benchmarks")
        pystats_dir = working_dir / "pystats"
        pystats_dir.mkdir()

        try:
            run_benchmarks(
                python,
                benchmark,
                extra_args=extra_args,
                env={"PYTHON_PYSTATS_DIR": "pystats/"},
                cwd=working_dir,
            )
        except NoBenchmarkError:
            pass
        else:
            if individual:
                run_summarize_stats(
                    pystats_dir, python, fork, ref, benchmark, False, flags=flags
                )

            for filename in pystats_dir.iterdir():
                os.rename(filename, Path(cumulative_stats_dir) / filename.name)

    finally:
        shutil.rmtree(working_dir)


def collect_pystats(
    python: Path,
    benchmarks: str,
    fork: str,
    ref: str,
    individual: bool,
    flags: Iterable[str] | None = None,
) -> None:
    all_benchmarks = get_benchmark_names(benchmarks)

    if flags is None:
        flags = []

    with tempfile.TemporaryDirectory() as tempdir:
        with multiprocessing.Pool() as pool:
            pool.starmap(
                _collect_single_pystats,
                [
                    (tempdir, python, benchmark, fork, ref, flags, individual)
                    for benchmark in all_benchmarks
                ],
            )

        if individual:
            benchmark_links = all_benchmarks
        else:
            benchmark_links = []

        run_summarize_stats(
            Path(tempdir), python, fork, ref, "all", True, benchmark_links, flags=flags
        )


def perf_to_csv(lines: Iterable[str], output: Path):
    event_count_prefix = "# Event count (approx.): "
    total = None

    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith(event_count_prefix):
            total = int(line[len(event_count_prefix) :].strip())
            continue
        if line.startswith("#") or line == "":
            continue
        if total is None:
            raise ValueError("Could not find total sample count")
        _, period, _, shared, _, symbol = line.split(maxsplit=5)
        self_time = float(int(period)) / total
        if self_time > 0.0:
            rows.append([self_time, 0.0, shared, symbol])

    rows.sort(key=itemgetter(0), reverse=True)

    with open(output, "w") as fd:
        csvwriter = csv.writer(fd)
        csvwriter.writerow(["self", "children", "object", "symbol"])
        for row in rows:
            csvwriter.writerow(row)


def collect_perf(python: Path, benchmarks: str):
    all_benchmarks = get_benchmark_names(benchmarks)

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
                        "-i",
                        "perf.data",
                    ],
                    encoding="utf-8",
                )
                perf_to_csv(
                    output.splitlines(), PROFILING_RESULTS / f"{benchmark}.perf.csv"
                )


def update_metadata(
    filename: Path,
    fork: str,
    ref: str,
    cpython: Path = Path("cpython"),
    run_id: str | None = None,
) -> None:
    with open(filename) as fd:
        content = ujson.load(fd)

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

    with open(filename, "w") as fd:
        ujson.dump(content, fd, indent=2)


def copy_to_directory(
    filename: Path, python: Path, fork: str, ref: str, flags: Iterable[str]
) -> None:
    result = Result.from_scratch(python, fork, ref, flags=flags)
    result.filename.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(filename, result.filename)


def run_summarize_stats(
    pystats_dir: Path,
    python: Path,
    fork: str,
    ref: str,
    benchmark: str,
    output_json: bool,
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

    print(f"RESULT: {result.filename}")

    args = [str(python), summarize_stats_path, pystats_dir]
    if output_json:
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

    with open(
        result.filename.with_suffix(".md"),
        "w",
    ) as fd:
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

    if output_json:
        if pystats_json.is_file():
            update_metadata(pystats_json, fork, ref)
        else:
            print(
                "WARNING: No pystats.json file generated. "
                "This is expected with CPython < 3.12"
            )


def get_excluded_benchmarks() -> list[str]:
    filename = Path("excluded_benchmarks.txt")
    if filename.is_file():
        with open(filename) as fd:
            return [x.strip() for x in fd.readlines()]
    return []


def select_benchmarks(benchmarks: str):
    if benchmarks == "all":
        return ",".join(["all", *[f"-{x}" for x in get_excluded_benchmarks() if x]])
    return benchmarks


def _main(
    mode: str,
    python: Path,
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
        update_metadata(Path("benchmark.json"), fork, ref, run_id=run_id)
        copy_to_directory(Path("benchmark.json"), python, fork, ref, flags)
    elif mode == "perf":
        collect_perf(python, benchmarks)
    elif mode == "pystats":
        collect_pystats(python, benchmarks, fork, ref, individual, flags)


def main():
    print("Environment variables:")
    for var in ENV_VARS:
        print(f"{var}={os.environ.get(var, '<unset>')}")

    parser = argparse.ArgumentParser(
        """
        Run benchmarks in `pyperformance` with the given python executable. Add
        additional metadata to a benchmark results file and then copy it to the
        correct location in the results tree.
        """
    )
    parser.add_argument(
        "mode", choices=["benchmark", "perf", "pystats"], help="The mode of execution"
    )
    parser.add_argument("python", help="The path to the Python executable")
    parser.add_argument("fork", help="The fork of CPython")
    parser.add_argument("ref", help="The git ref in the fork")
    parser.add_argument("benchmarks", help="The benchmarks to run")
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
    parser.add_argument(
        "--flag",
        action="append",
        help="Build or runtime flags",
    )
    args = parser.parse_args()

    if args.test_mode:
        import socket

        def gethostname():
            return "pyperf"

        socket.gethostname = gethostname

    _main(
        args.mode,
        Path(args.python),
        args.fork,
        args.ref,
        args.benchmarks,
        args.test_mode,
        args.run_id,
        args.individual,
        args.flag or [],
    )


if __name__ == "__main__":
    main()
