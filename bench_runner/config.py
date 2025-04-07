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
    filepath = Path(filepath)
    if filepath.is_file():
        with filepath.open("rb") as fd:
            return tomllib.load(fd)
    else:
        try:
            from pyodide.http import pyfetch  # type: ignore[import]
        except ImportError:
            raise RuntimeError("bench_runner.toml not found")
        toml_content = pyfetch(str(filepath)).string()
        if toml_content:
            return tomllib.loads(toml_content)
        else:
            raise RuntimeError("bench_runner.toml not found")
