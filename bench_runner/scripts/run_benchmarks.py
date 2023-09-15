from __future__ import annotations


import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Iterable, Optional, Union


from bench_runner import git
from bench_runner.result import Result


REPO_ROOT = Path()
BENCHMARK_JSON = REPO_ROOT / "benchmark.json"
PROFILING_RESULTS = REPO_ROOT / "profiling" / "results"
GITHUB_URL = "https://github.com/" + os.environ.get(
    "GITHUB_REPOSITORY", "faster-cpython/bench_runner"
)


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
    python: Union[Path, str],
    benchmarks: str,
    command_prefix: list[str] = [],
    test_mode: bool = False,
) -> None:
    if benchmarks.strip() == "":
        benchmarks = "all"

    if BENCHMARK_JSON.is_file():
        BENCHMARK_JSON.unlink()

    if test_mode:
        fast_arg = ["--fast"]
    else:
        fast_arg = []

    subprocess.call(
        command_prefix
        + [
            sys.executable,
            "-m",
            "pyperformance",
            "run",
        ]
        + fast_arg
        + [
            "-o",
            BENCHMARK_JSON,
            "--manifest",
            "benchmarks.manifest",
            "--benchmarks",
            benchmarks,
            "--python",
            python,
        ]
    )

    # pyperformance frequently returns an error if any of the benchmarks failed.
    # We only want to fail if things are worse than that.

    if not Path(BENCHMARK_JSON).is_file():
        raise NoBenchmarkError("No benchmark file created.")
    with open(BENCHMARK_JSON) as fd:
        contents = json.load(fd)
    if len(contents.get("benchmarks", [])) == 0:
        raise NoBenchmarkError("No benchmarks were run.")


def collect_pystats(
    python: Union[Path, str], benchmarks: str, fork: str, ref: str, individual: bool
) -> None:
    pystats_dir = Path("/tmp/py_stats")

    all_benchmarks = get_benchmark_names(benchmarks)

    with tempfile.TemporaryDirectory() as tempdir:
        for benchmark in all_benchmarks:
            try:
                run_benchmarks(python, benchmark)
            except NoBenchmarkError:
                pass
            else:
                if individual:
                    run_summarize_stats(python, fork, ref, benchmark, False)

            for filename in pystats_dir.iterdir():
                os.rename(filename, Path(tempdir) / filename.name)

        for filename in Path(tempdir).iterdir():
            os.rename(filename, pystats_dir / filename.name)

        if individual:
            benchmark_links = all_benchmarks
        else:
            benchmark_links = []

        run_summarize_stats(python, fork, ref, "all", True, benchmark_links)


def perf_to_csv(lines: Iterable[str], output: Path):
    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith("#") or line == "":
            continue
        children, self_time, _, shared, _, symbol = line.split(maxsplit=5)
        children = float(children[:-1])
        self = float(self_time[:-1])
        if children > 0.0 or self > 0.0:
            rows.append([self, children, shared, symbol])

    rows.sort(key=lambda x: x[0], reverse=True)

    with open(output, "w") as fd:
        csvwriter = csv.writer(fd)
        csvwriter.writerow(["self", "children", "object", "symbol"])
        for row in rows:
            csvwriter.writerow(row)


def collect_perf(python: Union[Path, str], benchmarks: str):
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
                    "--call-graph=dwarf",
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
    cpython="cpython",
    run_id: Optional[str] = None,
) -> None:
    with open(filename) as fd:
        content = json.load(fd)

    metadata = content.setdefault("metadata", {})

    metadata["commit_id"] = git.get_git_hash(cpython)
    metadata["commit_fork"] = fork
    metadata["commit_branch"] = ref
    metadata["commit_date"] = git.get_git_commit_date(cpython)
    if fork != "python" and ref != "main":
        merge_base = git.get_git_merge_base(cpython)
        if merge_base is not None:
            metadata["commit_merge_base"] = merge_base
    metadata["benchmark_hash"] = git.generate_combined_hash(
        ["pyperformance", "pyston-benchmarks"]
    )
    if run_id is not None:
        metadata["github_action_url"] = f"{GITHUB_URL}/actions/runs/{run_id}"

    with open(filename, "w") as fd:
        json.dump(content, fd, indent=2)


def copy_to_directory(filename: Path, python: str, fork: str, ref: str) -> None:
    result = Result.from_scratch(python, fork, ref)
    result.filename.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(filename, result.filename)


def run_summarize_stats(
    python: Union[Path, str],
    fork: str,
    ref: str,
    benchmark: str,
    output_json: bool,
    benchmarks: list[str] = [],
) -> None:
    summarize_stats_path = (
        Path(python).parent / "Tools" / "scripts" / "summarize_stats.py"
    )

    parts = ["pystats"]
    if benchmark != "all":
        parts.append(benchmark)
    result = Result.from_scratch(python, fork, ref, parts)
    result.filename.parent.mkdir(parents=True, exist_ok=True)
    pystats_json = result.filename.with_suffix(".json")

    args = [python, summarize_stats_path]
    if output_json:
        args.extend(["--json-output", pystats_json])

    table = subprocess.check_output(args, encoding="utf-8")

    header = textwrap.dedent(
        f"""
    # Pystats results

    - benchmark: {benchmark}
    - fork: {fork}
    - ref: {ref}
    - commit hash: {git.get_git_hash('cpython')[:7]}
    - commit date: {git.get_git_commit_date('cpython')}

    """
    )

    with open(
        result.filename.with_suffix(".md"),
        "w",
    ) as fd:
        fd.write(header)
        if benchmarks:
            for name in benchmarks:
                fd.write(
                    f"- [{name}]"
                    f"({Path(result.filename.name).with_suffix('')}-{name}.md)\n"
                )
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


def main(
    mode: str,
    python: str,
    fork: str,
    ref: str,
    benchmarks: str,
    test_mode: bool,
    run_id: Optional[str],
    individual: bool,
) -> None:
    if mode == "benchmark":
        run_benchmarks(python, benchmarks, [], test_mode)
        update_metadata(BENCHMARK_JSON, fork, ref, run_id=run_id)
        copy_to_directory(BENCHMARK_JSON, python, fork, ref)
    elif mode == "perf":
        collect_perf(python, benchmarks)
    elif mode == "pystats":
        collect_pystats(python, benchmarks, fork, ref, individual)


if __name__ == "__main__":
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
    args = parser.parse_args()

    if args.test_mode:
        import socket

        def gethostname():
            return "pyperf"

        socket.gethostname = gethostname

    main(
        args.mode,
        args.python,
        args.fork,
        args.ref,
        args.benchmarks,
        args.test_mode,
        args.run_id,
        args.individual,
    )
