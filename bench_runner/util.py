import functools
import hashlib
import os
from pathlib import Path


from . import config


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


def apply_suffix(path: Path, suffix: str) -> Path:
    """
    Like Path.with_suffix but allows adding things like "-mem.svg".
    """
    return path.parent / (path.stem + suffix)


@functools.cache
def get_excluded_benchmarks() -> set[str]:
    conf = config.get_bench_runner_config()
    return set(conf.get("benchmarks", {}).get("excluded", []))
