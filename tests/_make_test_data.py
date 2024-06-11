#!/usr/bin/env python

from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


import rich


def create_benchmarking_dir(root):
    rich.print("Checking out CPython")
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

    rich.print("Checking out python-macrobenchmarks")
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

    rich.print("Checking out pyperformance")
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

    rich.print("Creating venv")
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
    shutil.copyfile(
        Path(__file__).parents[1] / "pyproject.toml", root / "pyproject.toml"
    )

    return root


def create_benchmarking_tarball():
    output_filename = Path(__file__).parent / "data" / "benchmarking.tar.gz"
    with tempfile.TemporaryDirectory() as tmpdir:
        create_benchmarking_dir(Path(tmpdir))
        rich.print("Creating tarball")
        if output_filename.exists():
            output_filename.unlink()
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(Path(tmpdir), arcname=".")


if __name__ == "__main__":
    create_benchmarking_tarball()
