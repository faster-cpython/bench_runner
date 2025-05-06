from __future__ import annotations


import argparse
from collections import defaultdict
import datetime
import functools
from operator import attrgetter
from pathlib import Path
from typing import Iterable, Sequence, TypeAlias


import rich
import rich.console
import rich.prompt
import rich.table
import rich_argparse


from bench_runner import config
from bench_runner import flags as mflags
from bench_runner import gh
from bench_runner import git
from bench_runner import result as mod_result
from bench_runner import runners
from bench_runner.util import PathLike


RunnerType: TypeAlias = runners.Runner


@functools.cache
def _get_hash_and_date(cpython: PathLike, ref: str) -> tuple[str, datetime.datetime]:
    hash, date = git.get_log("%H %cI", cpython, ref).split()
    return hash, datetime.datetime.fromisoformat(date)


class Commit:
    """
    Represents a single commit to possibly benchmark.
    """

    def __init__(self, cpython: PathLike, ref: str, source: str):
        self.cpython = cpython
        self.ref = ref
        self.source = source
        self.runners: list[RunnerType] = []

    def _set_hash_and_date(self) -> None:
        self._hash, self._date = _get_hash_and_date(self.cpython, self.ref)

    @property
    def hash(self) -> str:
        if not hasattr(self, "_hash"):
            self._set_hash_and_date()
        return self._hash

    @property
    def date(self) -> datetime.datetime:
        if not hasattr(self, "_date"):
            self._set_hash_and_date()
        return self._date


def get_all_with_prefix(
    cpython: PathLike, tags: Iterable[str], prefix: str
) -> Iterable[Commit]:
    """
    Get all tags with the given prefix.
    """
    for tag in tags:
        if tag.startswith(prefix):
            yield Commit(cpython, tag, f"--all-with-prefix {prefix}")


def get_latest_with_prefix(
    cpython: PathLike, tags: Iterable[str], prefix: str
) -> Iterable[Commit]:
    """
    Get the most recent (by commit date) tag with the given prefix.
    """
    commits = []
    for tag in tags:
        if tag.startswith(prefix):
            commits.append(Commit(cpython, tag, f"--latest-with-prefix {prefix}"))

    commits.sort(key=attrgetter("date"))
    if len(commits) > 0:
        yield commits[-1]


def next_weekday(d: datetime.datetime, weekday: int) -> datetime.datetime:
    """
    Given datetime `d`, returns the next date on the given ISO weekday.
    """
    days_ahead = weekday - d.weekday()
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return d + datetime.timedelta(days_ahead)


def get_weekly_since(cpython: PathLike, start_date: str) -> Iterable[Commit]:
    """
    Get weekly commits on Sundays since the given start date.
    """
    start = datetime.datetime.fromisoformat(start_date).replace(
        tzinfo=datetime.timezone.utc
    )
    today = datetime.datetime.now().replace(tzinfo=datetime.timezone.utc)

    commits = git.get_log(
        "%cI %h", cpython, n=0, extra=[f"--since={start_date}"]
    ).splitlines()
    commits.sort()
    commits = [x.split() for x in commits]
    commits = [(datetime.datetime.fromisoformat(x), y) for x, y in commits]

    current_date = next_weekday(start, 7)
    while current_date < today and len(commits):
        while len(commits):
            commit_date, ref = commits.pop(0)
            if commit_date > current_date:
                yield Commit(cpython, ref, f"--weekly-since {start_date}")
                current_date = next_weekday(current_date, 7)
                break


def get_bisect(cpython: PathLike, refs: Sequence[str]) -> Iterable[Commit]:
    if len(refs) != 2:
        raise ValueError(f"Each --bisect entry must contain 2 refs, got {len(refs)}")

    yield Commit(
        cpython,
        git.bisect_commits(cpython, refs[0], refs[1]),
        f"--bisect {refs[0]} {refs[1]}",
    )


def match_machine(a: str, b: str) -> bool:
    return (
        (a == "amd64" and b == "x86_64") or (a == "x86_64" and b == "amd64") or (a == b)
    )


def remove_existing(
    force: bool,
    commits: Iterable[Commit],
    runners: Sequence[RunnerType],
    results: Sequence[mod_result.Result] | None = None,
) -> Iterable[Commit]:
    """
    Remove any runner/commit combinations that we already have results for.
    If force is True, all runner/commit combinations will be generated.
    """
    if force:
        for commit in commits:
            commit.runners = list(runners)
            yield commit
        return

    if results is None:
        results = mod_result.load_all_results(
            None, Path("results"), sorted=False, match=False
        )

    for commit in commits:
        commit.runners = []
        for runner in runners:
            for result in results:
                if result.nickname == runner.nickname and commit.hash.startswith(
                    result.cpython_hash
                ):
                    break
            else:
                commit.runners.append(runner)

        if len(commit.runners):
            yield commit


def get_commits(
    cpython: PathLike,
    tags: Iterable[str],
    all_with_prefix: Iterable[str],
    latest_with_prefix: Iterable[str],
    weekly_since: Iterable[str],
    bisect: Iterable[Sequence[str]],
) -> Iterable[Commit]:
    for entry in all_with_prefix:
        yield from get_all_with_prefix(cpython, tags, entry)

    for entry in latest_with_prefix:
        yield from get_latest_with_prefix(cpython, tags, entry)

    for entry in weekly_since:
        yield from get_weekly_since(cpython, entry)

    for entry in bisect:
        yield from get_bisect(cpython, entry)


def deduplicate_commits(
    cpython: PathLike, commits: Iterable[Commit]
) -> Iterable[Commit]:
    commits_by_hash = defaultdict(list)

    for commit in commits:
        commits_by_hash[commit.hash].append(commit)

    for commit_set in commits_by_hash.values():
        first_commit = commit_set[0]
        if len(commit_set) == 1:
            yield first_commit
        else:
            yield Commit(
                cpython, first_commit.ref, ", ".join(x.source for x in commit_set)
            )


def format_runners(
    active_runners: Sequence[RunnerType], all_runners: Sequence[RunnerType]
):
    result = []
    for runner in all_runners:
        if runner in active_runners:
            result.append("X")
        else:
            result.append("-")
    return "".join(result)


def _main(
    cpython: PathLike,
    all_with_prefix: Sequence[str] | None,
    latest_with_prefix: Sequence[str] | None,
    weekly_since: Sequence[str] | None,
    bisect: Sequence[Sequence[str]] | None,
    runners: Sequence[RunnerType],
    force: bool,
    flags: list[str],
    all_runners: Sequence[RunnerType],
) -> None:
    all_with_prefix = all_with_prefix or []
    latest_with_prefix = latest_with_prefix or []
    weekly_since = weekly_since or []
    bisect = bisect or []

    tags = git.get_tags(cpython)

    commits = get_commits(
        cpython, tags, all_with_prefix, latest_with_prefix, weekly_since, bisect
    )

    commits = deduplicate_commits(cpython, commits)
    commits = remove_existing(force, commits, runners)

    commits = sorted(commits, key=attrgetter("date"))

    rich.print(f"runners: {', '.join(f'[blue]{x.nickname}[/blue]' for x in runners)}")
    rich.print()

    table = rich.table.Table(title="Benchmark runs")
    table.add_column("date")
    table.add_column("hash")
    table.add_column("ref")
    table.add_column("machine")
    table.add_column("source")

    runs = 0
    for commit in commits:
        table.add_row(
            str(commit.date)[:10],
            commit.hash[:7],
            commit.ref[:15],
            format_runners(commit.runners, runners),
            commit.source,
        )
        runs += len(commit.runners)

    console = rich.console.Console()
    console.print(table)

    rich.print(f"Selected {len(commits)} commits, {runs} runs.")
    if runs == 0:
        return

    if rich.prompt.Confirm.ask("Are you sure you want to run them all?", default=False):
        for commit in commits:
            if len(commit.runners) == len(all_runners):
                gh.benchmark(ref=commit.hash, machine="all")
            else:
                for runner in commit.runners:
                    gh.benchmark(
                        ref=commit.hash,
                        machine=runner.name,
                        flags=flags,
                    )


def main():
    cfg = config.get_config()
    all_runners = [x for x in cfg.runners.values() if x.available]
    runners_by_names = {x.nickname: x for x in all_runners}

    parser = argparse.ArgumentParser(
        description="""
        Fire off a set of benchmark jobs based on tags in the cpython
        repository. Useful for regenerating or catching up with old data. The
        set of tags to run will be displayed for confirmation before actually
        setting up the jobs.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "--all-with-prefix",
        action="append",
        help="Add all tags with the given version prefix, e.g. v3.11",
    )
    parser.add_argument(
        "--latest-with-prefix",
        action="append",
        help="Add the latest tag with the given version prefix, e.g. v3.10",
    )
    parser.add_argument(
        "--weekly-since",
        action="append",
        help="Select one commit per week since the given iso date, e.g. 2022-09-01",
    )
    parser.add_argument(
        "--bisect",
        nargs=2,
        action="append",
        help="Select the commit between the two given refs",
    )
    parser.add_argument(
        "--machine",
        choices=[*runners_by_names.keys(), "all"],
        default=all_runners[0].nickname,
        help="The machine to run on.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run benchmark, even if we already have results for that commit hash.",
    )
    parser.add_argument(
        "--flags",
        help=(
            "A comma-separated value of configuration flags, "
            f"from {', '.join(flag.gha_variable for flag in mflags.FLAGS)}"
        ),
    )
    parser.add_argument("cpython", type=Path, help="The path to a checkout of CPython")

    args = parser.parse_args()

    if args.machine != "all":
        use_runners = [runners_by_names[args.machine]]
    else:
        use_runners = all_runners

    _main(
        args.cpython,
        args.all_with_prefix,
        args.latest_with_prefix,
        args.weekly_since,
        args.bisect,
        use_runners,
        args.force,
        mflags.parse_flags(args.flags),
        all_runners,
    )


if __name__ == "__main__":
    main()
