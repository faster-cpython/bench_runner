from __future__ import annotations


import dataclasses
import os
import socket
from typing import Literal


from .util import PathLike


@dataclasses.dataclass
class PlotConfig:
    # The name of the runner in the plot legend
    name: str
    # A matplotlib color to use for this runner in plots
    color: str = "C0"
    # A matplotlib line style to use for this runner in plots
    style: str = "-"
    # A matplotlib marker to use for this runner in plots
    marker: str = "s"


@dataclasses.dataclass
class Runner:
    # The short "nickname" of the runner
    nickname: str
    # The OS of the runner
    os: Literal["linux", "darwin", "windows", "unknown"]
    # The architecture of the runner, e.g. "x86_64", "arm64"
    arch: str
    # The hostname of the runner, used to identify which runner we are running on
    hostname: str
    # Whether the runner is available for benchmarking (e.g. not a VM)
    available: bool = True
    # Environment variables to set for the benchmark
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    # The name of the Github runner to use for this machine, only required when
    # this runner needs to map to another physical machine.
    github_runner_name: str | None = None
    # Whether to include this runner in the "all" choice
    include_in_all: bool = True
    # The plot configuration for this runner
    plot: PlotConfig | None = None
    # The number of cores to use to compile CPython. If not provided, `make -j`
    # will be used.
    use_cores: int | None = None

    def __post_init__(self):
        if self.github_runner_name is None:
            self.github_runner_name = self.name
        if self.plot is None:
            self.plot = PlotConfig(name=self.nickname)
        else:
            self.plot = PlotConfig(**self.plot)  # pyright: ignore[reportCallIssue]

    @property
    def name(self) -> str:
        return f"{self.os}-{self.arch}-{self.nickname}"

    @property
    def display_name(self) -> str:
        return f"{self.os} {self.arch} ({self.nickname})"


unknown_runner = Runner("unknown", "unknown", "unknown", "unknown", False, {}, None)


def get_runners_by_hostname(cfgpath: PathLike | None = None) -> dict[str, Runner]:
    from . import config

    return {x.hostname: x for x in config.get_config(cfgpath).runners.values()}


def get_nickname_for_hostname(
    hostname: str | None = None, cfgpath: PathLike | None = None
) -> str:
    # The envvar BENCHMARK_MACHINE_NICKNAME is used to override the machine that
    # results are reported for.
    if "BENCHMARK_MACHINE_NICKNAME" in os.environ:
        return os.environ["BENCHMARK_MACHINE_NICKNAME"]
    return get_runner_for_hostname(hostname, cfgpath).nickname


def get_runner_by_nickname(nickname: str, cfgpath: PathLike | None = None) -> Runner:
    from . import config

    return config.get_config(cfgpath).runners.get(nickname, unknown_runner)


def get_runner_for_hostname(
    hostname: str | None = None, cfgpath: PathLike | None = None
) -> Runner:
    if hostname is None:
        hostname = socket.gethostname()
    return get_runners_by_hostname(cfgpath).get(hostname, unknown_runner)
