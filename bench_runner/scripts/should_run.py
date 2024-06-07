# Determines if this should run.
# If force is `true`, we always run, otherwise, we only run if we don't have
# results.

import argparse
from pathlib import Path
import subprocess
import sys


import rich_argparse


# NOTE: This file should import in Python 3.9 or later so it can at least print
# the error message that the version of Python is too old.


def _main(
    force: bool,
    fork: str,
    ref: str,
    machine: str,
    pystats: bool,
    flag_str: str,
    cpython: Path = Path("cpython"),
    results_dir: Path = Path("results"),
) -> None:
    if sys.version_info[:2] < (3, 11):
        print(
            "The benchmarking infrastructure requires Python 3.11 or later.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Now that we've assert we are Python 3.11 or later, we can import
    # parts of our library.
    from bench_runner import flags as mflags
    from bench_runner import git
    from bench_runner.result import has_result
    from bench_runner import util

    flags = mflags.parse_flags(flag_str)

    if "PYTHON_UOPS" in flags and "JIT" in flags:
        print("Tier 2 interpreter and JIT may not be selected at the same time")
        sys.exit(1)

    try:
        commit_hash = git.get_git_hash(cpython)
    except subprocess.CalledProcessError:
        # This will fail if the cpython checkout failed for some reason. Print
        # a nice error message since the one the checkout itself gives is
        # totally inscrutable.
        print("The checkout of cpython failed.", file=sys.stderr)
        print(f"You specified fork {fork!r} and ref {ref!r}.", file=sys.stderr)
        print("Are you sure you entered the fork and ref correctly?", file=sys.stderr)
        # Fail the rest of the workflow
        sys.exit(1)

    found_result = has_result(
        results_dir,
        commit_hash,
        machine,
        pystats,
        flags,
        util.get_benchmark_hash(),
        progress=False,
    )

    if force:
        if found_result is not None:
            for filepath in found_result.filename.parent.iterdir():
                if filepath.suffix != ".json":
                    git.remove(results_dir.parent, filepath)
        should_run = True
    else:
        should_run = machine == "all" or found_result is None

    print(f"should_run={str(should_run).lower()}")


def main():
    parser = argparse.ArgumentParser(
        description="Do we need to run this commit?",
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "force",
        help="If true, force a re-run",
    )
    parser.add_argument("fork")
    parser.add_argument("ref")
    parser.add_argument("machine")
    parser.add_argument("pystats")
    parser.add_argument("flags")
    args = parser.parse_args()

    _main(
        args.force != "false",
        args.fork,
        args.ref,
        args.machine,
        args.pystats != "false",
        args.flags,
    )


if __name__ == "__main__":
    main()
