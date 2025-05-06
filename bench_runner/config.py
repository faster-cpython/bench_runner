"""
Handles the loading of the bench_runner.toml configuration file.
"""

import dataclasses
import functools
from pathlib import Path
import tomllib


from . import flags as mflags
from . import plot as mplot
from . import runners as mrunners
from .util import PathLike


@dataclasses.dataclass
class Bases:
    # The base versions to compare every benchmark run to.
    # Should be a full-specified version, e.g. "3.13.0".
    versions: list[str]
    # List of configuration flags that are compared against the default build of
    # its commit merge base.
    compare_to_default: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if len(self.versions) == 0:
            raise RuntimeError(
                "No `bases.versions` are defined in `bench_runner.toml`. "
            )
        mflags.normalize_flags(self.compare_to_default)


@dataclasses.dataclass
class Notify:
    # The Github issue to use to send notification emails
    notification_issue: int = 0


@dataclasses.dataclass
class PublishMirror:
    # Whether to skip publishing to the mirror
    skip: bool = False


@dataclasses.dataclass
class Benchmarks:
    # Benchmarks to exclude from plots.
    excluded_benchmarks: set[str] = dataclasses.field(default_factory=set)

    def __post_init__(self):
        self.excluded_benchmarks = set(self.excluded_benchmarks)


@dataclasses.dataclass
class Weekly:
    flags: list[str] = dataclasses.field(default_factory=list)
    runners: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self.flags = mflags.normalize_flags(self.flags)


@dataclasses.dataclass
class Config:
    bases: Bases
    runners: dict[str, mrunners.Runner]
    publish_mirror: PublishMirror = dataclasses.field(default_factory=PublishMirror)
    benchmarks: Benchmarks = dataclasses.field(default_factory=Benchmarks)
    notify: Notify = dataclasses.field(default_factory=Notify)
    longitudinal_plot: mplot.LongitudinalPlotConfig | None = None
    flag_effect_plot: mplot.FlagEffectPlotConfig | None = None
    benchmark_longitudinal_plot: mplot.BenchmarkLongitudinalPlotConfig | None = None
    weekly: dict[str, Weekly] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.bases = Bases(**self.bases)  # pyright: ignore[reportCallIssue]
        if len(self.runners) == 0:
            raise RuntimeError(
                "No runners are defined in `bench_runner.toml`. "
                "Please set up some runners first."
            )
        self.runners = {
            name: mrunners.Runner(
                nickname=name, **runner  # pyright: ignore[reportCallIssue]
            )
            for name, runner in self.runners.items()
        }
        if isinstance(self.publish_mirror, dict):
            self.publish_mirror = PublishMirror(**self.publish_mirror)
        if isinstance(self.benchmarks, dict):
            self.benchmarks = Benchmarks(**self.benchmarks)
        if isinstance(self.notify, dict):
            self.notify = Notify(**self.notify)
        if isinstance(self.longitudinal_plot, dict):
            self.longitudinal_plot = mplot.LongitudinalPlotConfig(
                **self.longitudinal_plot
            )
        if isinstance(self.flag_effect_plot, dict):
            self.flag_effect_plot = mplot.FlagEffectPlotConfig(**self.flag_effect_plot)
        if isinstance(self.benchmark_longitudinal_plot, dict):
            self.benchmark_longitudinal_plot = mplot.BenchmarkLongitudinalPlotConfig(
                **self.benchmark_longitudinal_plot
            )
        if len(self.weekly) == 0:
            self.weekly = {"default": Weekly(runners=list(self.runners.keys()))}
        else:
            self.weekly = {
                name: Weekly(**weekly)  # pyright: ignore[reportCallIssue]
                for name, weekly in self.weekly.items()
            }


@functools.cache
def get_config(filepath: PathLike | None = None) -> Config:
    if filepath is None:
        filepath = Path("bench_runner.toml")
    else:
        filepath = Path(filepath)

    with filepath.open("rb") as fd:
        content = tomllib.load(fd)

    return Config(**content)
