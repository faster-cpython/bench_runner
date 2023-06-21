import configparser
import functools
from pathlib import Path
from typing import Dict, List


class Runner:
    def __init__(
        self, nickname: str, os: str, arch: str, hostname: str, available: bool
    ):
        self.nickname = nickname
        self.os = os
        self.arch = arch
        self.hostname = hostname
        self.available = available

    @property
    def name(self) -> str:
        return f"{self.os}-{self.arch}-{self.nickname}"


@functools.cache
def get_runners(path=Path("runners.ini")) -> List[Runner]:
    config = configparser.ConfigParser()
    config.read(path)
    runners = []
    for nickname in config.sections():
        section = config[nickname]
        runners.append(
            Runner(
                nickname,
                section["os"],
                section["arch"],
                section["hostname"],
                section.getboolean("available", True),
            )
        )

    assert len(runners)

    return runners


def get_runners_by_hostname() -> Dict[str, Runner]:
    return {x.hostname: x for x in get_runners()}


def get_runners_by_nickname() -> Dict[str, Runner]:
    return {x.nickname: x for x in get_runners()}


def get_nickname_for_hostname(hostname: str) -> str:
    return get_runners_by_hostname()[hostname].nickname


def get_runner_by_nickname(nickname: str) -> Runner:
    return get_runners_by_nickname()[nickname]
