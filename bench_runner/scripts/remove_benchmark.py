from __future__ import annotations


import argparse
import json
from pathlib import Path
from typing import Sequence


import rich
import rich_argparse


from bench_runner import result
from bench_runner.scripts import generate_results
from bench_runner import util
from bench_runner.util import PathLike


def remove_benchmark(
    filename: PathLike, remove: set[str], keep_hash: set[str], dry_run: bool
):
    with Path(filename).open() as fd:
        data = json.load(fd)

    if data["metadata"]["benchmark_hash"] in keep_hash:
        return

    benchmarks = []
    for benchmark in data["benchmarks"]:
        if "metadata" in benchmark:
            name = benchmark["metadata"]["name"]
        else:
            name = data["metadata"]["name"]
        if name not in remove:
            benchmarks.append(benchmark)

    data["benchmarks"] = benchmarks

    if not dry_run:
        with Path(filename).open("w") as fd:
            json.dump(data, fd, indent=2)


def _main(benchmarks: Sequence[str], keep_hash: Sequence[str], dry_run: bool = False):
    rich.print(f"Removing benchmarks {', '.join(benchmarks)} from all results")

    keep_hash_set = set(keep_hash)
    benchmarks_set = set(benchmarks)

    if not dry_run:
        if Path("longitudinal.json").is_file():
            Path("longitudinal.json").unlink()

    for filename in util.track(list(Path("results").glob("**/*")), "Deleting results"):
        if filename.is_dir():
            continue
        if filename.name != "README.md":
            res = result.Result.from_filename(filename)
            if res.result_info[0] == "raw results":
                remove_benchmark(filename, benchmarks_set, keep_hash_set, dry_run)

    rich.print("Regenerating all derived results. This will take quite some time...")

    if not dry_run:
        generate_results._main(Path(), force=True)


def main():
    parser = argparse.ArgumentParser(
        description="Remove one or more benchmarks from the entire dataset",
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )

    parser.add_argument(
        "benchmark",
        nargs="+",
        help="Benchmark to remove",
    )
    parser.add_argument(
        "--keep-hash", action="append", help="The benchmark hash(es) to leave alone"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    if args.keep_hash is None:
        keep_hash = []
    else:
        keep_hash = args.keep_hash

    _main(args.benchmark, keep_hash, args.dry_run)


if __name__ == "__main__":
    main()
