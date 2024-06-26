from __future__ import annotations


import configparser
import functools
import os
from pathlib import Path


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
def get_runners(path: Path | None = None) -> list[Runner]:
    if path is None:
        path = Path("runners.ini")

    config = configparser.ConfigParser()
    # Don't convert keys to lowercase
    config.optionxform = str  # type: ignore
    config.read(path)
    runners = []
    for nickname in config.sections():
        section = config[nickname]

        envvars = {
            key[4:]: val for key, val in section.items() if key.startswith("env-")
        }

        runners.append(
            Runner(
                nickname,
                section["os"],
                section["arch"],
                section["hostname"],
                section.getboolean("available", True),
                envvars,
                section.get("github-runner-name"),
            )
        )

    if len(runners) == 0:
        raise RuntimeError(f"No runners are defined in `{path}`. Please set up some runners first.")

    return runners


def get_runners_by_hostname() -> dict[str, Runner]:
    return {x.hostname: x for x in get_runners()}


def get_runners_by_nickname() -> dict[str, Runner]:
    return {x.nickname: x for x in get_runners()}


def get_nickname_for_hostname(hostname: str) -> str:
    # The envvar BENCHMARK_MACHINE_NICKNAME is used to override the machine that
    # results are reported for.
    if "BENCHMARK_MACHINE_NICKNAME" in os.environ:
        return os.environ["BENCHMARK_MACHINE_NICKNAME"]
    return get_runners_by_hostname()[hostname].nickname


def get_runner_by_nickname(nickname: str) -> Runner:
    return get_runners_by_nickname()[nickname]
