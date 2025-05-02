from __future__ import annotations


import collections
import functools
import os
import socket


from . import config
from .util import PathLike


class PlotConfig:
    def __init__(
        self, name: str, color: str = "C0", style: str = "-", marker: str = "s"
    ):
        self.name = name
        self.color = color
        self.style = style
        self.marker = marker


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
        include_in_all: bool = True,
        plot: dict[str, str] | None = None,
        tags: list[str] | None = None,
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
        self.include_in_all = include_in_all
        if plot is None:
            plot = {"name": nickname}
        self.plot = PlotConfig(**plot)
        self.tags = tags

    @property
    def name(self) -> str:
        return f"{self.os}-{self.arch}-{self.nickname}"

    @property
    def display_name(self) -> str:
        return f"{self.os} {self.arch} ({self.nickname})"


unknown_runner = Runner("unknown", "unknown", "unknown", "unknown", False, {}, None)


@functools.cache
def get_runners(cfgpath: PathLike | None = None) -> list[Runner]:
    conf = config.get_bench_runner_config(cfgpath).get("runners", {})
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
                section.get("include_in_all", True),
                section.get("plot", None),
                section.get("tags"),
            )
        )

    if len(runners) == 0:
        raise RuntimeError(
            "No runners are defined in `bench_runner.toml`. "
            "Please set up some runners first."
        )

    return runners


def get_runners_by_hostname(cfgpath: PathLike | None = None) -> dict[str, Runner]:
    return {x.hostname: x for x in get_runners(cfgpath)}


def get_runners_by_nickname(cfgpath: PathLike | None = None) -> dict[str, Runner]:
    return {x.nickname: x for x in get_runners(cfgpath)}


def get_nickname_for_hostname(
    hostname: str | None = None, cfgpath: PathLike | None = None
) -> str:
    # The envvar BENCHMARK_MACHINE_NICKNAME is used to override the machine that
    # results are reported for.
    if "BENCHMARK_MACHINE_NICKNAME" in os.environ:
        return os.environ["BENCHMARK_MACHINE_NICKNAME"]
    return get_runner_for_hostname(hostname, cfgpath).nickname


def get_runner_by_nickname(nickname: str, cfgpath: PathLike | None = None) -> Runner:
    return get_runners_by_nickname(cfgpath).get(nickname, unknown_runner)


def get_runner_for_hostname(
    hostname: str | None = None, cfgpath: PathLike | None = None
) -> Runner:
    if hostname is None:
        hostname = socket.gethostname()
    return get_runners_by_hostname(cfgpath).get(hostname, unknown_runner)


def get_tags(cfgpath: PathLike | None = None) -> dict[str, list[Runner]]:
    d = collections.defaultdict(list)
    for runner in get_runners(cfgpath):
        if runner.tags:
            for tag in runner.tags:
                d[tag].append(runner)
    return dict(d)


def get_runners_from_nicknames_and_tags(
    nicknames: list[str], cfgpath: PathLike | None = None
) -> list[Runner]:
    result = []
    tags = get_tags(cfgpath)
    runners = get_runners_by_nickname(cfgpath)
    for nickname in nicknames:
        if nickname.startswith("tag "):
            tag = nickname.removeprefix("tag ")
            if tag not in tags:
                raise ValueError(f"Tag {tag} not found in bench_runner.toml")
            result.extend(tags[nickname.removeprefix("tag ")])
        else:
            if nickname not in runners:
                raise ValueError(f"Runner {nickname} not found in bench_runner.toml")
            result.append(runners[nickname])
    return result
