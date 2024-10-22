from __future__ import annotations


import argparse
import datetime
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Sequence


import rich
import rich.progress
import rich_argparse


from bench_runner.bases import get_bases, get_minimum_version_for_all_comparisons
from bench_runner.result import load_all_results
from bench_runner.scripts.generate_results import _main as generate_results
from bench_runner.util import PathLike


def dir_size(path: PathLike) -> int:
    total = 0
    for root, _, files in os.walk(path):
        root = Path(root)
        for file in files:
            total += os.stat(root / file).st_size
    return total


def _main(
    repo_dir: PathLike, days: int, dry_run: bool, bases: Sequence[str] | None = None
):
    results_dir = Path(repo_dir) / "results"
    if bases is None:
        bases = get_bases()
    if len(bases) == 0:
        raise ValueError("Must have at least one base specified")

    results = load_all_results(bases, results_dir, sorted=False, match=False)
    all_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    keep_dirs = set()
    remove_generated_files = set()

    earliest = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()

    for result in rich.progress.track(
        results, description="Selecting results to remove"
    ):
        if result.result_info[0] != "raw results":
            continue
        if (
            not (result.version.endswith("+") or result.version.endswith("a0"))
            or result.version in bases
            or result.run_date >= earliest
        ):
            keep_dirs.add(result.filename.parent)
            if (
                result.version not in bases
                and result.parsed_version.release[0:2]
                < get_minimum_version_for_all_comparisons()
            ):
                remove_generated_files.update(result.filename.parent.glob("*-vs-*"))

    for filename in results_dir.glob("**/*"):
        if m := re.match(".*-vs-(.+?)(-.+)?$", filename.stem):
            base = m.group(1)
            if base != "base" and base not in bases:
                remove_generated_files.add(filename)

    rich.print(
        f"Removing {len(all_dirs) - len(keep_dirs)} of {len(all_dirs)} "
        "results directories"
    )
    rich.print(f"Removing {len(remove_generated_files)} comparison files")

    total = 0
    for d in rich.progress.track(all_dirs, "Removing directories"):
        if d not in keep_dirs:
            rich.print(f"Removing {d}")
            total += dir_size(d)
            if not dry_run:
                shutil.rmtree(d)

    for f in rich.progress.track(remove_generated_files, "Removing comparison files"):
        if f.is_file():
            rich.print(f"Removing {f}")
            total += f.stat().st_size
            if not dry_run:
                f.unlink()

    rich.print(f"Saved {total:,} bytes")

    if not dry_run:
        rich.print("Regenerating results...")

        generate_results(repo_dir, force=False, bases=bases)


def main():
    parser = argparse.ArgumentParser(
        description="""
        Purge old results that aren't associated with an exact tag.
        Should be run every few months to keep the repository size under control.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )

    parser.add_argument(
        "repo_dir",
        nargs="?",
        type=Path,
        default=Path(),
        help="The location of the results repository",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="The number of days to retain",
    )
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not args.repo_dir.is_dir():
        print(f"{args.repo_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    _main(args.repo_dir, args.days, args.dry_run)


if __name__ == "__main__":
    main()
