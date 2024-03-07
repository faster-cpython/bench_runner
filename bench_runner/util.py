import hashlib
import os


def status(char: str) -> None:
    print(char, end="", flush=True)


TIER2_FLAGS = ["PYTHON_UOPS"]
JIT_FLAGS = ["JIT"]


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
