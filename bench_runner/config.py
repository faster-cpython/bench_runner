"""
Handles the loading of the bench_runner.toml configuration file.
"""

import functools
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore


@functools.cache
def get_bench_runner_config(
    filepath: Path | str = Path("bench_runner.toml"),
):
    with Path(filepath).open("rb") as fd:
        return tomllib.load(fd)
