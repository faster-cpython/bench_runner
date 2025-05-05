"""
Utility to compare a matrix of arbitrary commits, for when the usual key
versions or git merge base aren't enough.
"""

from __future__ import annotations


import argparse
from pathlib import Path
import sys
from typing import Literal, TextIO, Iterable, Sequence, TypeAlias


import rich
import rich_argparse


from bench_runner import config
from bench_runner import flags as mflags
from bench_runner import result as mod_result
from bench_runner import util
from bench_runner.util import PathLike


ParsedCommits: TypeAlias = Sequence[
    tuple[str, list[str], str, Iterable[mod_result.Result]]
]  # (commit_hash, flags, name, results)


def parse_commit(commit: str) -> tuple[str, str, list[str]]:
    if "," in commit:
        result = commit.split(",", 1)
        commit, name = result
    else:
        name = commit
    suffix = commit[-1]
    flags = []
    if suffix not in "0123456789abcdefABCDEF":
        for flag_descr in mflags.FLAGS:
            if flag_descr.short_name[0] == suffix:
                commit = commit[:-1]
                flags = [flag_descr.name]
                break
    return (commit, name, flags)


def get_machines(results: Iterable[mod_result.Result]) -> set[str]:
    return set(result.nickname for result in results)


def compare_pair(
    output_dir: PathLike,
    machine: str | None,
    ref_name: str,
    ref: mod_result.Result,
    head_name: str,
    head: mod_result.Result,
    counter: list[int],
) -> str:
    output_dir = Path(output_dir)

    rich.print(
        f"Comparing {counter[0]+1: 2}/{counter[1]: 2}: {head_name} vs. {ref_name}"
    )
    counter[0] += 1

    name_parts = []
    if machine is not None:
        name_parts.append(machine)
    name_parts.extend([head_name, "vs", ref_name])
    name = "-".join(name_parts)
    comparison = mod_result.BenchmarkComparison(ref, head, "base", True)
    entry = [comparison.summary]
    for func, suffix, file_type in comparison.get_files():
        output_filename = util.apply_suffix(output_dir / name, suffix)
        func(output_filename)
        entry.append(f"[{util.TYPE_TO_ICON[file_type]}]({output_filename.name})")

    return "".join(entry)


def write_row(fd: TextIO, columns: Iterable[str]):
    fd.write(f"| {' | '.join(columns)} |\n")


def name_and_hash(name: str, hash: str) -> str:
    if name == hash:
        return name
    return f"{name} ({hash})"


def get_first_result_for_machine(
    results: Iterable[mod_result.Result], machine: str | None
) -> mod_result.Result:
    return next(
        result for result in results if machine is None or result.nickname == machine
    )


def do_one_to_many(
    fd: TextIO,
    parsed_commits: ParsedCommits,
    machine: str | None,
    output_dir: PathLike,
    counter: list[int],
) -> None:
    _, _, first_name, first_results = parsed_commits[0]
    first_result = get_first_result_for_machine(first_results, machine)
    write_row(fd, ["commit", "change"])
    write_row(fd, ["--"] * 2)
    for hash, _, name, results in parsed_commits[1:]:
        result = get_first_result_for_machine(results, machine)
        link = compare_pair(
            output_dir, machine, first_name, first_result, name, result, counter
        )
        write_row(fd, [name_and_hash(name, hash), link])


def do_many_to_many(
    fd,
    parsed_commits: ParsedCommits,
    machine: str | None,
    output_dir: PathLike,
    counter: list[int],
) -> None:
    write_row(fd, ["", *[name_and_hash(x[2], x[0]) for x in parsed_commits]])
    write_row(fd, ["--"] * (len(parsed_commits) + 1))
    for hash1, flags1, name1, results1 in parsed_commits:
        columns = [name1]
        result1 = get_first_result_for_machine(results1, machine)
        for hash2, flags2, name2, results2 in parsed_commits:
            if hash1 == hash2 and flags1 == flags2:
                columns.append("")
            else:
                result2 = get_first_result_for_machine(results2, machine)
                link = compare_pair(
                    output_dir, machine, name1, result1, name2, result2, counter
                )
                columns.append(link)
        write_row(fd, columns)

    fd.write("\n\nRows are 'bases', columns are 'heads'\n")


def _main_with_hashes(
    commits: Sequence[str], output_dir: Path, comparison_type: Literal["1:n", "n:n"]
):
    cfg = config.get_config()

    results = mod_result.load_all_results(
        None, Path("results"), sorted=False, match=False
    )

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

    if "cloud" in machines:
        machines.remove("cloud")

    if len(machines) == 0:
        raise ValueError("No single machine in common with all of the results")

    match comparison_type:
        case "1:n":
            total = (len(parsed_commits) - 1) * len(machines)
            func = do_one_to_many
        case "n:n":
            total = ((len(parsed_commits) ** 2) - len(parsed_commits)) * len(machines)
            func = do_many_to_many
        case _:
            raise ValueError(f"Unknown comparison type {comparison_type}")

    counter = [0, total]
    with (output_dir / "README.md").open("w", encoding="utf-8") as fd:
        for machine in machines:
            fd.write(f"# {cfg.runners[machine].display_name}\n\n")
            func(fd, parsed_commits, machine, output_dir, counter)
            fd.write("\n")
    rich.print()


def _main_with_files(commits: Sequence[str], output_dir: Path, comparison_type: str):
    parsed_results = []
    for commit in commits:
        if "," in commit:
            commit_path, name = commit.split(",", 1)
            commit_path = Path(commit_path)
        else:
            commit_path = Path(commit)
            name = commit_path.stem
        parsed_results.append(
            (name, [], name, [mod_result.Result.from_arbitrary_filename(commit_path)])
        )

    match comparison_type:
        case "1:n":
            total = len(commits) - 1
            func = do_one_to_many
        case "n:n":
            total = (len(commits) ** 2) - len(commits)
            func = do_many_to_many
        case _:
            raise ValueError(f"Unknown comparison type {comparison_type}")

    counter = [0, total]
    with (output_dir / "README.md").open("w", encoding="utf-8") as fd:
        fd.write("# Comparisons\n\n")
        func(fd, parsed_results, None, output_dir, counter)
        fd.write("\n")
    rich.print()


def _main(
    commits: Sequence[str], output_dir: PathLike, comparison_type: Literal["1:n", "n:n"]
):
    if len(commits) < 2:
        raise ValueError("Must provide at least 2 commits")

    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir()

    if all(commit.endswith(".json") for commit in commits):
        _main_with_files(commits, output_dir_path, comparison_type)
    elif any(commit.endswith(".json") for commit in commits):
        raise ValueError("All commits must be either hashes or JSON files")
    else:
        _main_with_hashes(commits, output_dir_path, comparison_type)


def main():
    parser = argparse.ArgumentParser(
        description="""
        Generate a set of comparisons between arbitrary commits. The commits
        must already exist in the dataset.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )

    parser.add_argument(
        "--output-dir", required=True, help="Directory to output results to."
    )
    parser.add_argument(
        "commit",
        nargs="+",
        help="""
            Commits or files to compare. If ends with ".json", it is a path to a
            JSON file, otherwise, it is a git commit hash prefix. May optionally
            have a friendly name after a comma, e.g. c0ffee,main.  If ends with
            a "T", use the Tier 2 run for that commit. If ends with a "J", use
            the JIT run for that commit.  If ends with a "N", use the NOGIL run
            for that commit.
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
        rich.print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
