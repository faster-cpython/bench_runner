import functools
import hashlib
import os
from pathlib import Path
from typing import TypeAlias, Union


import numpy as np
from numpy.typing import NDArray
from pyperf import _utils


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


def is_significant(
    sample1: NDArray[np.float64], sample2: NDArray[np.float64]
) -> tuple[float, float]:
    # This is a port of pyperf._utils.is_significant that uses Numpy and is
    # therefore faster.
    @functools.lru_cache(None)
    def tdist95conf_level(deg_freedom: float) -> float:
        return _utils.tdist95conf_level(deg_freedom)

    if len(sample1) != len(sample2):
        raise ValueError("different number of values")

    deg_freedom = float(len(sample1) * 2 - 2)
    critical_value = tdist95conf_level(deg_freedom)

    mean1 = sample1.mean()
    squares1 = (sample1 - mean1) ** 2
    mean2 = sample2.mean()
    squares2 = (sample2 - mean2) ** 2
    error = (np.sum(squares1) + np.sum(squares2) / deg_freedom) / float(len(sample1))

    diff = sample1.mean() - sample2.mean()
    t_score = diff / np.sqrt(error * 2)

    return (abs(t_score) >= critical_value, t_score)
