import functools
import itertools
import os
from pathlib import Path
import shutil
import subprocess
from typing import TypeAlias, Union


from . import config


PathLike: TypeAlias = Union[str, os.PathLike]


TYPE_TO_ICON = {
    "table": "📄",
    "time plot": "📈",
    "memory plot": "🧠",
}


def apply_suffix(path: PathLike, suffix: str) -> Path:
    """
    Like Path.with_suffix but allows adding things like "-mem.svg".
    """
    path_ = Path(path)
    return path_.parent / (path_.stem + suffix)


@functools.cache
def get_excluded_benchmarks() -> set[str]:
    conf = config.get_bench_runner_config()
    benchmarks_section = conf.get("benchmarks", {})
    for key in ("excluded", "excluded_benchmarks"):
        if key in benchmarks_section:
            return set(benchmarks_section[key])
    return set()


def has_any_element(iterable):
    """
    Checks if an iterable (like a generator) has at least one element
    without consuming the original iterable more than necessary.
    """
    first, iterable = itertools.tee(iterable, 2)  # Create two independent iterators
    try:
        next(first)  # Try to get the first element
        return True  # If successful, the generator is not empty
    except StopIteration:
        return False  # If StopIteration is raised, the generator is empty


def safe_which(cmd: str) -> str:
    """
    shutil, but raises a RuntimeError if the command is not found.
    """
    path = shutil.which(cmd)
    if path is None:
        raise RuntimeError(f"Command {cmd} not found in PATH")
    return path


def get_brew_prefix(command: str) -> str:
    """
    Get the prefix of the Homebrew installation.
    """
    try:
        prefix = subprocess.check_output(["brew", "--prefix", command])
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Unable to find brew installation prefix for {command}")
    return prefix.decode("utf-8").strip()
