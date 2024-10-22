"""
Handles the loading of the bench_runner.toml configuration file.
"""

import functools
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

from .util import PathLike


@functools.cache
def get_bench_runner_config(
    filepath: PathLike = Path("bench_runner.toml"),
):
    with Path(filepath).open("rb") as fd:
        return tomllib.load(fd)
