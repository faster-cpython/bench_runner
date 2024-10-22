import functools
import hashlib
import os
from pathlib import Path
from typing import TypeAlias, Union


from . import config


PathLike: TypeAlias = Union[str, os.PathLike]


def get_benchmark_hash() -> str:
    hash = hashlib.sha256()
    hash.update(os.environ["PYPERFORMANCE_HASH"].encode("ascii")[:7])
    hash.update(os.environ["PYSTON_BENCHMARKS_HASH"].encode("ascii")[:7])
    return hash.hexdigest()[:6]


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
