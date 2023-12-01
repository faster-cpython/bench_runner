from __future__ import annotations


import argparse
import datetime
from pathlib import Path
import shutil
import sys
from typing import Optional


from bench_runner.bases import get_bases
from bench_runner.result import load_all_results
from bench_runner.scripts.generate_results import _main as generate_results


def _main(repo_dir: Path, days: int, dry_run: bool, bases: Optional[list[str]] = None):
    results_dir = repo_dir / "results"
    if bases is None:
        bases = get_bases()
    if len(bases) == 0:
        raise ValueError("Must have at least one base specified")

    print("Loading results")
    results = load_all_results(bases, results_dir)
    all_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    keep_dirs = set()

    earliest = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()

    for result in results:
        if result.result_info[0] != "raw results":
            continue
        if (
            not (result.version.endswith("+") or result.version.endswith("a0"))
            or result.version in bases
            or result.run_date >= earliest
        ):
            keep_dirs.add(result.filename.parent)

    print(
        f"Removing {len(all_dirs) - len(keep_dirs)} of {len(all_dirs)} "
        "results directories"
    )

    for d in all_dirs:
        if d not in keep_dirs:
            print(f"Removing {d}")
            if not dry_run:
                shutil.rmtree(d)

    if not dry_run:
        print("Regenerating results")

        generate_results(repo_dir, force=False, bases=bases)


def main():
    parser = argparse.ArgumentParser(
        "Purge old results that aren't associated with an exact tag. "
        "Should be run every few months to keep the repository size under control. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
