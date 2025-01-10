from __future__ import annotations


import functools
import os


from . import config
from functools import lru_cache


class Runner:
    def __init__(
        self,
        nickname: str,
        os: str,
        arch: str,
        hostname: str,
        available: bool,
        env: dict[str, str],
        # Override the Github self-hosted runner name if different from
        # os-arch-nickname
        github_runner_name: str | None,
    ):
        self.nickname = nickname
        self.os = os
        self.arch = arch
        self.hostname = hostname
        self.available = available
        self.env = env
        if github_runner_name is None:
            github_runner_name = self.name
        self.github_runner_name = github_runner_name

    @property
    def name(self) -> str:
        return f"{self.os}-{self.arch}-{self.nickname}"

    @property
    def display_name(self) -> str:
        return f"{self.os} {self.arch} ({self.nickname})"


@functools.cache
def get_runners() -> list[Runner]:
    conf = config.get_bench_runner_config().get("runners", [{}])[0]
    runners = []
    for nickname, section in conf.items():
        runners.append(
            Runner(
                nickname,
                section["os"],
                section["arch"],
                section["hostname"],
                section.get("available", True),
                section.get("env", {}),
                section.get("github_runner_name"),
            )
        )

    if len(runners) == 0:
        raise RuntimeError(
            "No runners are defined in `bench_runner.toml`. "
            "Please set up some runners first."
        )

    return runners


@lru_cache(maxsize=1)  # Cache this function to avoid re-computation
def get_runners_by_hostname() -> dict[str, Runner]:
    return {x.hostname: x for x in get_runners()}


def get_runners_by_nickname() -> dict[str, Runner]:
    return {x.nickname: x for x in get_runners()}


def get_nickname_for_hostname(hostname: str) -> str:
    # The envvar BENCHMARK_MACHINE_NICKNAME is used to override the machine that
    # results are reported for.
    nickname = os.environ.get("BENCHMARK_MACHINE_NICKNAME")
    if nickname is not None:
        return nickname
    runners_by_hostname = get_runners_by_hostname()
    return (
        runners_by_hostname[hostname].nickname
        if hostname in runners_by_hostname
        else ""
    )


def get_runner_by_nickname(nickname: str) -> Runner:
    return get_runners_by_nickname()[nickname]
