"""
Handles the loading of the bench_runner.toml configuration file.
"""

import functools
from pathlib import Path
import tomllib
from typing import Any


from . import runners
from .util import PathLike


@functools.cache
def get_bench_runner_config(filepath: PathLike | None = None):
    if filepath is None:
        filepath = Path("bench_runner.toml")
    else:
        filepath = Path(filepath)

    with filepath.open("rb") as fd:
        return tomllib.load(fd)


def get_config_for_current_runner(filepath: PathLike | None = None) -> dict[str, Any]:
    config = get_bench_runner_config(filepath)
    runner = runners.get_runner_for_hostname(cfgpath=filepath)
    all_runners = config.get("runners", {})
    if len(all_runners) >= 1:
        return all_runners.get(runner.nickname, {})
    return {}
