"""
Handles the loading of the bench_runner.toml configuration file.
"""

import functools
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore


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
    return config.get("runners", {}).get(runner.nickname, {})
