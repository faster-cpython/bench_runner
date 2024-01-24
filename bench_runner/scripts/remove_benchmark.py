from __future__ import annotations


import argparse
import json
from pathlib import Path


from bench_runner import result
from bench_runner.scripts import generate_results


def remove_benchmark(filename: Path, remove: list[str]):
    with open(filename) as fd:
        data = json.load(fd)

    benchmarks = []
    for benchmark in data["benchmarks"]:
        if "metadata" in benchmark:
            name = benchmark["metadata"]["name"]
        else:
            name = data["metadata"]["name"]
        if name not in remove:
            benchmarks.append(benchmark)

    data["benchmarks"] = benchmarks

    with open(filename, "w") as fd:
        json.dump(data, fd, indent=2)


def _main(benchmarks: list[str], dry_run: bool = False):
    print(f"Removing benchmarks {benchmarks} from all results")
    for filename in Path("results").glob("**/*"):
        print(".", end="")
        if filename.is_dir():
            continue
        if filename.name == "README.md":
            if not dry_run:
                filename.unlink()
        else:
            res = result.Result.from_filename(filename)
            match res.result_info[0]:
                case "raw results":
                    if not dry_run:
                        remove_benchmark(filename, benchmarks)
                case "plot" | "table":
                    if not dry_run:
                        filename.unlink()
                case _:
                    pass
    print()

    print("Regenerating all derived results. This will take quite some time.")

    generate_results._main(Path())


def main():
    parser = argparse.ArgumentParser(
        "Remove one or more benchmarks from the entire dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    _main(args.benchmark, args.dry_run)


if __name__ == "__main__":
    main()
