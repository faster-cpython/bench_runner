import hashlib
import os
from pathlib import Path


def status(char: str) -> None:
    print(char, end="", flush=True)


TIER2_FLAGS = ["PYTHON_UOPS"]
JIT_FLAGS = ["JIT"]


def get_flags(tier2, jit):
    flags = []
    if tier2:
        flags.extend(TIER2_FLAGS)
    if jit:
        flags.extend(JIT_FLAGS)
    return flags


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
    Like Path.with_suffix but allows adding things like "-mem.png".
    """
    return path.parent / (path.stem + suffix)
