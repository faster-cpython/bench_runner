import filecmp
import os
from pathlib import Path
import tarfile


import pytest


os.environ["PYPERFORMANCE_HASH"] = "f7f36509e2e81e9a20cfeadddd6608f2378ff26c"
os.environ["PYSTON_BENCHMARKS_HASH"] = "d4868ff7825f3996e0005197643ed56eba4fb567"


DATA_PATH = Path(__file__).parent / "data"


@pytest.fixture
def benchmarks_checkout(tmp_path):
    with tarfile.open(DATA_PATH / "benchmarking.tar.gz") as tar:
        tar.extractall(tmp_path)

    if not filecmp.cmp(
        tmp_path / "pyproject.toml", Path(__file__).parents[1] / "pyproject.toml"
    ):
        raise AssertionError(
            "Dependencies have changed. "
            "Re-run the _make_test_data.py script and commit the result."
        )

    return tmp_path
