from __future__ import annotations


import argparse
import json
from pathlib import Path


from bench_runner import result
from bench_runner.scripts import generate_results
from bench_runner import util


def remove_benchmark(filename: Path, remove: list[str], benchmark_hash: str):
    with open(filename) as fd:
        data = json.load(fd)

    if data["metadata"]["benchmark_hash"] == benchmark_hash:
        util.status("/")
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

    util.status(".", end="")

    with open(filename, "w") as fd:
        json.dump(data, fd, indent=2)


def _main(benchmarks: list[str], benchmark_hash: str, dry_run: bool = False):
    print(f"Removing benchmarks {benchmarks} from all results")

    if not dry_run:
        if Path("longitudinal.json").is_file():
            Path("longitudinal.json").unlink()

    for filename in Path("results").glob("**/*"):
        if filename.is_dir():
            continue
        if filename.name != "README.md":
            res = result.Result.from_filename(filename)
            if res.result_info[0] == "raw results":
                if not dry_run:
                    remove_benchmark(filename, benchmarks, benchmark_hash)
    print()

    print("Regenerating all derived results. This will take quite some time.")

    generate_results._main(Path(), force=True)


def main():
    parser = argparse.ArgumentParser(
        "Remove one or more benchmarks from the entire dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "benchmark_hash", type=str, help="The benchmark hash to leave alone"
    )
    parser.add_argument(
        "benchmark",
        nargs="+",
        help="Benchmark to remove",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    _main(args.benchmark, args.benchmark_hash, args.dry_run)


if __name__ == "__main__":
    main()
