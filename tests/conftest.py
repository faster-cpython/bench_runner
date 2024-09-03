import os
from pathlib import Path
import shutil
import subprocess
import sys


from filelock import FileLock
import pytest


os.environ["PYPERFORMANCE_HASH"] = "f7f36509e2e81e9a20cfeadddd6608f2378ff26c"
os.environ["PYSTON_BENCHMARKS_HASH"] = "d4868ff7825f3996e0005197643ed56eba4fb567"


DATA_PATH = Path(__file__).parent / "data"


ROOT = None


def _setup_repositories(root):
    root.mkdir()

    subprocess.check_call(
        [
            "git",
            "clone",
            "https://github.com/python/cpython",
            "--depth",
            "1",
            "--branch",
            "v3.10.4",
        ],
        cwd=root,
    )

    subprocess.check_call(
        [
            "git",
            "clone",
            "https://github.com/mdboom/python-macrobenchmarks",
            "--depth",
            "1",
            "--branch",
            "benchmarking-test",
            "pyston-benchmarks",
        ],
        cwd=root,
    )

    subprocess.check_call(
        [
            "git",
            "clone",
            "https://github.com/mdboom/pyperformance",
            "--depth",
            "1",
            "--branch",
            "benchmarking-test",
        ],
        cwd=root,
    )

    venv_dir = root / "venv"
    venv_python = venv_dir / "bin" / "python"
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir], cwd=root)
    subprocess.check_call([venv_python, "-m", "pip", "install", "setuptools"], cwd=root)
    subprocess.check_call(
        [venv_python, "-m", "pip", "install", root / "pyperformance"], cwd=root
    )
    subprocess.check_call(
        [
            root / "venv" / "bin" / "python",
            "-m",
            "pip",
            "install",
            Path(__file__).parents[1],
        ],
        cwd=root,
    )


@pytest.fixture(scope="session")
def setup_repositories(tmp_path_factory, worker_id):
    # get the temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    root = root_tmp_dir / "benchmarking"

    if worker_id == "master":
        if root.exists():
            shutil.rmtree(root)
        _setup_repositories(root)
    else:
        with FileLock(str(root_tmp_dir / "file.lock")):
            if not root.exists():
                _setup_repositories(root)

    return root


@pytest.fixture
def benchmarks_checkout(setup_repositories, tmp_path):
    for path in setup_repositories.iterdir():
        if path.is_dir():
            shutil.copytree(path, tmp_path / path.name)
        elif path.is_file():
            shutil.copyfile(path, tmp_path / path.name)
    return tmp_path
