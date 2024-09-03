"""
Handles the loading of the bench_runner.toml configuration file.
"""

import functools
from pathlib import Path
import tomllib


@functools.cache
def get_bench_runner_config(
    filepath: Path = Path("bench_runner.toml"),
):
    with open(filepath, "rb") as fd:
        return tomllib.load(fd)
