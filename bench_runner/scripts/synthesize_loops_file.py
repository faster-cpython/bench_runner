import argparse
import collections
import errno
import json
import pathlib
import sys
from typing import Iterable

import rich_argparse


def parse_result(results_file, benchmark_data):
    with results_file.open() as f:
        result = json.load(f)
    bms = result["benchmarks"]
    if len(bms) == 1 and "metadata" not in bms[0]:
        # Sometimes a .json file contains just a single benchmark.
        bms = [result]
    for bm in bms:
        if "metadata" not in bm:
            raise RuntimeError(f"Invalid data {bm.keys()!r} in {results_file}")
            return
        benchmark_data[bm["metadata"]["name"]].append(bm["metadata"]["loops"])


def _main(
    loops_file: pathlib.Path,
    update: bool,
    overwrite: bool,
    merger: str,
    results: Iterable[pathlib.Path],
):
    if not update and not overwrite and loops_file.exists():
        raise OSError(
            errno.EEXIST,
            f"{loops_file} exists (use -f to overwrite, -u to merge data)",
        )
    if update and merger in ("median", "mean"):
        print(
            f"WARNING: merging existing data with {merger!r} "
            + "overrepresents new results",
            file=sys.stderr,
        )
    benchmark_data = collections.defaultdict(list)
    if update:
        parse_result(loops_file, benchmark_data)
    for result_file in results:
        parse_result(result_file, benchmark_data)

    merge_func = {
        "max": max,
        "min": min,
        # The only merge strategy that may not produce one of the input
        # values, and probably a bad idea to use.
        "mean": lambda L: int(round(sum(L) / len(L))),
        # Close enough to median for benchmarking work.
        "median": lambda L: L[len(L) // 2],
    }[merger]

    # pyperformance expects a specific layout, and needs the top-level
    # metadata even if it's empty.
    loops_data = {"benchmarks": [], "metadata": {}}
    for bm in sorted(benchmark_data):
        loops = merge_func(benchmark_data[bm])
        bm_result = {"metadata": {"name": bm, "loops": loops}}
        loops_data["benchmarks"].append(bm_result)
    with loops_file.open("w") as f:
        json.dump(loops_data, f, sort_keys=True, indent=4)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="""
        Synthesize a loops.json file for use with `pyperformance`'s
        `--same-loops` (or `PYPERFORMANCE_LOOPS_FILE`) from one or more
        benchmark results.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-o", "--loops_file", help="loops file to write to", required=True
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-u", "--update", action="store_true", help="add to existing loops file"
    )
    group.add_argument(
        "-f", "--overwrite", action="store_true", help="replace loops file"
    )
    parser.add_argument(
        "-s",
        "--select",
        choices=("max", "min", "median", "mean"),
        default="max",
        help="how to merge multiple runs",
    )
    parser.add_argument("results", nargs="+", help="benchmark results to parse")
    args = parser.parse_args()

    _main(
        pathlib.Path(args.loops_file),
        args.update,
        args.overwrite,
        args.select,
        [pathlib.Path(r) for r in args.results],
    )


if __name__ == "__main__":
    main()
