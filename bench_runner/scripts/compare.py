"""
Utility to compare a matrix of arbitrary commits, for when the usual key
versions or git merge base aren't enough.
"""

from __future__ import annotations


import argparse
from pathlib import Path
import sys
from typing import Iterable


from bench_runner import result as mod_result
from bench_runner import plot
from bench_runner import runners as mod_runners
from bench_runner import util


def parse_commit(commit: str) -> tuple[str, str, list[str]]:
    if "," in commit:
        result = commit.split(",", 1)
        commit, name = result
    else:
        name = commit
    if commit.endswith("T"):
        commit = commit[:-1]
        flags = util.TIER2_FLAGS
    elif commit.endswith("J"):
        commit = commit[:-1]
        flags = util.JIT_FLAGS
    else:
        flags = []
    return (commit, name, flags)


def get_machines(results: Iterable[mod_result.Result]) -> set[str]:
    return set(result.nickname for result in results)


def compare_pair(
    output_dir: Path,
    machine: str,
    ref_name: str,
    ref: mod_result.Result,
    head_name: str,
    head: mod_result.Result,
    counter: list[int],
) -> str:
    print(f"Comparing {counter[0]+1}/{counter[1]}", end="\r")
    counter[0] += 1

    name = f"{machine}-{head_name}-vs-{ref_name}"
    comparison = mod_result.BenchmarkComparison(ref, head, "")
    if comparison.contents is None:
        raise RuntimeError()
    with open(output_dir / f"{name}.md", "w", encoding="utf-8") as fd:
        fd.write(comparison.contents)
    compare = mod_result.BenchmarkComparison(ref, head, "base")
    plot.plot_diff(
        compare.get_timing_diff(),
        output_dir / f"{name}.png",
        f"{head_name} vs. {ref_name}",
        ("slower", "faster"),
    )

    return f"{comparison.summary} [table]({name}.md) [plot]({name}.png)"


def write_row(fd, columns: list[str]):
    fd.write(f"| {' | '.join(columns)} |\n")


def do_one_to_many(
    fd,
    parsed_commits: list[tuple[str, list[str], str, list[mod_result.Result]]],
    machine: str,
    output_dir: Path,
    counter: list[int],
) -> None:
    _, _, first_name, first_results = parsed_commits[0]
    first_result = next(
        result for result in first_results if result.nickname == machine
    )
    write_row(fd, ["commit", "change"])
    write_row(fd, ["--"] * 2)
    for hash, _, name, results in parsed_commits[1:]:
        result = next(result for result in results if result.nickname == machine)
        link = compare_pair(
            output_dir, machine, first_name, first_result, name, result, counter
        )
        write_row(fd, [f"{name} ({hash})", link])


def do_many_to_many(
    fd,
    parsed_commits: list[tuple[str, list[str], str, list[mod_result.Result]]],
    machine: str,
    output_dir: Path,
    counter: list[int],
) -> None:
    write_row(fd, ["", *[f"{x[2]} ({x[0]})" for x in parsed_commits]])
    write_row(fd, ["--"] * (len(parsed_commits) + 1))
    for hash1, flags1, name1, results1 in parsed_commits:
        columns = [name1]
        result1 = next(result for result in results1 if result.nickname == machine)
        for hash2, flags2, name2, results2 in parsed_commits:
            if hash1 == hash2 and flags1 == flags2:
                columns.append("")
            else:
                result2 = next(
                    result for result in results2 if result.nickname == machine
                )
                link = compare_pair(
                    output_dir, machine, name1, result1, name2, result2, counter
                )
                columns.append(link)
        write_row(fd, columns)

    fd.write("\n\nRows are 'bases', columns are 'heads'\n")


def _main(commits: list[str], output_dir: Path, comparison_type: str):
    results = mod_result.load_all_results(
        None, Path("results"), sorted=False, match=False
    )

    if len(commits) < 2:
        raise ValueError("Must provide at least 2 commits")

    parsed_commits = []
    machines = set()

    for commit in commits:
        commit_hash, name, flags = parse_commit(commit)

        subresults = [
            result
            for result in results
            if result.cpython_hash.startswith(commit_hash) and result.flags == flags
        ]

        if len(subresults) == 0:
            raise ValueError(f"Couldn't find commit {commit_hash}")

        parsed_commits.append((commit_hash, flags, name, subresults))

        if len(machines) == 0:
            machines = get_machines(subresults)
        else:
            machines &= get_machines(subresults)

    if len(machines) == 0:
        raise ValueError("No single machine in common with all of the results")

    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir()

    match comparison_type:
        case "1:n":
            total = (len(parsed_commits) - 1) * len(machines)
            func = do_one_to_many
        case "n:n":
            total = ((len(parsed_commits) ** 2) - len(parsed_commits)) * len(machines)
            func = do_many_to_many
        case _:
            raise ValueError(f"Unknown comparison type {comparison_type}")

    runners = mod_runners.get_runners_by_nickname()

    counter = [0, total]
    with open(output_dir_path / "README.md", "w", encoding="utf-8") as fd:
        for machine in machines:
            fd.write(f"# {runners[machine].display_name}\n\n")
            func(fd, parsed_commits, machine, output_dir_path, counter)
            fd.write("\n")
    print()


def main():
    parser = argparse.ArgumentParser(
        """
        Generate a set of comparisons between arbitrary commits. The commits
        must already exist in the dataset.
        """
    )

    parser.add_argument(
        "--output-dir", required=True, help="Directory to output results to."
    )
    parser.add_argument(
        "commit",
        nargs="+",
        help="""
            Commits to compare. Must be a git commit hash prefix. May optionally
            have a friendly name after a comma, e.g. c0ffee,main.  If ends with
            a "T", use the Tier 2 run for that commit. If ends with a "J", use the
            JIT run for that commit.
        """,
    )
    parser.add_argument(
        "--type",
        choices=["1:n", "n:n"],
        default="1:n",
        help="""
            Compare the first commit to all others, or do the full product of all
            commits
        """,
    )

    args = parser.parse_args()

    try:
        _main(args.commit, Path(args.output_dir), args.type)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
