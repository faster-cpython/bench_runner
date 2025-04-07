# Various git-related utilities
from __future__ import annotations


import datetime
from pathlib import Path
import subprocess


from .util import PathLike
from .util import rich_print


def get_log(
    format: str,
    dirname: PathLike,
    ref: str | None = None,
    n: int = 1,
    extra: list[str] | None = None,
) -> str:
    """
    format: The git pretty format string for each log entry
    dirname: Local checkout of the repository
    ref: If provided, the git ref to show
    n: If < 1, show full log, otherwise the number of entries to show
    extra: Extra arguments to pass to `git log`
    """
    if extra is None:
        extra = []

    ref_args = [] if ref is None else [ref]
    n_args = [] if n < 1 else ["-n", str(n)]

    return subprocess.check_output(
        ["git", "log", f"--pretty=format:{format}", *n_args, *ref_args, *extra],
        encoding="utf-8",
        cwd=dirname,
    ).strip()


def get_git_hash(dirname: PathLike) -> str:
    return get_log("%h", dirname)


def get_git_commit_date(dirname: PathLike) -> str:
    return get_log("%cI", dirname)


def remove(repodir: Path, path: PathLike) -> None:
    subprocess.check_output(
        ["git", "rm", str(path)],
        cwd=repodir,
    )


def get_git_merge_base(dirname: PathLike) -> str | None:
    # We need to make sure we have commits from main that are old enough to be
    # the base of this branch, but not so old that we waste a ton of bandwidth
    commit_date = datetime.datetime.fromisoformat(get_git_commit_date(dirname))
    commit_date = commit_date - datetime.timedelta(days=365 * 2)

    # Get current commit hash
    commit_hash = get_log("%H", dirname)

    try:
        subprocess.check_call(
            [
                "git",
                "remote",
                "add",
                "upstream",
                "https://github.com/python/cpython.git",
            ],
            cwd=dirname,
        )
    except subprocess.CalledProcessError as e:
        if e.returncode not in (3, 128):
            raise

    subprocess.check_call(
        [
            "git",
            "fetch",
            "upstream",
            "main",
            "--shallow-since",
            commit_date.isoformat(),
        ],
        cwd=dirname,
    )

    try:
        merge_base = subprocess.check_output(
            ["git", "merge-base", "upstream/main", "HEAD"],
            cwd=dirname,
            encoding="utf-8",
        ).strip()
    except subprocess.CalledProcessError:
        rich_print("[red]Failed to get merge base[/red]")
        return None

    if merge_base == commit_hash:
        # Get the parent commit if the merge base is the same as the current commit
        return get_log("%H", dirname, "HEAD^")

    return merge_base


def get_tags(dirname: PathLike) -> list[str]:
    subprocess.check_call(["git", "fetch", "--tags"], cwd=dirname)
    return subprocess.check_output(
        ["git", "tag"], cwd=dirname, encoding="utf-8"
    ).splitlines()


def get_commits_between(dirname: PathLike, ref1: str, ref2: str) -> list[str]:
    return list(
        subprocess.check_output(
            ["git", "rev-list", "--ancestry-path", f"{ref1}..{ref2}"],
            cwd=dirname,
            encoding="utf-8",
        ).splitlines()
    )


def bisect_commits(dirname: PathLike, ref1: str, ref2: str) -> str:
    commits = get_commits_between(dirname, ref1, ref2)
    return commits[len(commits) // 2]
