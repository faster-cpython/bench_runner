"""
Handles the loading of the bench_runner.toml configuration file.
"""

import functools
from pathlib import Path
import tomllib
from typing import Any


from . import runners


@functools.cache
def get_bench_runner_config(
    filepath: Path | str = Path("bench_runner.toml"),
):
    with Path(filepath).open("rb") as fd:
        return tomllib.load(fd)


def get_config_for_current_runner() -> dict[str, Any]:
    config = get_bench_runner_config()
    runner = runners.get_runner_for_hostname()
    all_runners = config.get("runners", [])
    if len(all_runners) >= 1:
        return all_runners[0].get(runner.nickname, {})
    return {}
