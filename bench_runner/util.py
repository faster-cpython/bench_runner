import contextlib
import functools
import itertools
import os
from pathlib import Path
import shutil
import subprocess
import stat
import sys
from typing import Iterable, Iterator, Literal, TypeAlias, Union


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


@functools.cache
def get_simple_platform() -> Literal["linux", "macos", "windows"]:
    """
    Return a basic platform name: linux, macos or windows.
    """
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform == "darwin":
        return "macos"
    elif sys.platform.startswith("win"):
        return "windows"
    raise RuntimeError(f"Unsupported platform {sys.platform}.")


def format_seconds(value: float) -> str:
    """
    Given a float value in seconds, formats it into a human-readable string with
    the appropriate precision.
    """
    _TIMEDELTA_UNITS = ("sec", "ms", "us", "ns")

    for i in range(2, -9, -1):
        if value >= 10.0**i:
            break
    else:
        i = -9

    precision = 2 - i % 3
    k = -(i // 3) if i < 0 else 0
    factor = 10 ** (k * 3)
    unit = _TIMEDELTA_UNITS[k]
    return f"{value * factor:.{precision}f} {unit}"


def valid_version(version: str) -> bool:
    from packaging import version as pkg_version

    try:
        pkg_version.parse(version)
        return True
    except pkg_version.InvalidVersion:
        return False


if sys.platform.startswith("win"):
    if sys.version_info >= (3, 12):

        def smart_rmtree(path: PathLike) -> None:
            def onexc(func, path, exc):
                # Is the error an access error?
                if not os.access(path, os.W_OK):
                    os.chmod(path, stat.S_IWUSR)
                    func(path)
                else:
                    raise exc

            shutil.rmtree(path, onexc=onexc)

    else:

        def smart_rmtree(path: PathLike) -> None:
            def onerror(func, path, exc_info):
                # Is the error an access error?
                if not os.access(path, os.W_OK):
                    os.chmod(path, stat.S_IWUSR)
                    func(path)
                else:
                    raise exc_info[1]

            shutil.rmtree(path, onerror=onerror)

else:
    smart_rmtree = shutil.rmtree


if os.getenv("GITHUB_ACTIONS") == "true":

    @contextlib.contextmanager
    def log_group(text: str) -> Iterator:
        print(f"::group::{text}", file=sys.stderr)
        try:
            yield
        finally:
            print("::endgroup::", file=sys.stderr)

    def track(iterable: Iterable, name: str) -> Iterable:
        with log_group(name):
            yield from iterable

else:
    try:
        import rich
    except ImportError:

        @contextlib.contextmanager
        def log_group(text: str) -> Iterator:
            print(text)
            yield

        def track(iterable: Iterable, name: str) -> Iterable:
            print(name)
            return iterable

    else:

        @contextlib.contextmanager
        def log_group(text: str) -> Iterator:
            rich.print(f"[b]{text}:[/b]")
            yield

        def track(iterable: Iterable, name: str) -> Iterable:
            from rich.progress import track

            return track(iterable, description=name)
