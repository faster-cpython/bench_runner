import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


from bench_runner import benchmark_definitions
from bench_runner import git
from bench_runner.scripts import run_benchmarks


DATA_PATH = Path(__file__).parent / "data"


def dont_get_git_merge_base(monkeypatch):
    def dummy(*args, **kwargs):
        return None

    monkeypatch.setattr(git, "get_git_merge_base", dummy)


def hardcode_benchmark_hash(monkeypatch):
    def dummy(*args, **kwargs):
        return "215d35"

    monkeypatch.setattr(benchmark_definitions, "get_benchmark_hash", dummy)


def test_update_metadata(benchmarks_checkout, monkeypatch):
    dont_get_git_merge_base(monkeypatch)
    hardcode_benchmark_hash(monkeypatch)

    shutil.copy(
        DATA_PATH
        / "results"
        / "bm-20211208-3.11.0a3-2e91dba"
        / "bm-20211208-linux-x86_64-python-main-3.11.0a3-2e91dba.json",
        benchmarks_checkout / "benchmarks.json",
    )
    run_benchmarks.update_metadata(
        benchmarks_checkout / "benchmarks.json",
        "myfork",
        "myref",
        benchmarks_checkout / "cpython",
        "12345",
    )

    with open(benchmarks_checkout / "benchmarks.json") as fd:
        content = json.load(fd)

    metadata = content["metadata"]

    assert metadata["commit_id"] == "9d38120"
    assert metadata["commit_fork"] == "myfork"
    assert metadata["commit_branch"] == "myref"
    assert metadata["commit_date"].startswith("2022-03-23T20:12:04")
    assert "commit_merge_base" not in metadata
    assert metadata["benchmark_hash"] == "215d35"
    assert (
        metadata["github_action_url"]
        == "https://github.com/faster-cpython/bench_runner/actions/runs/12345"
    )


def test_run_benchmarks(benchmarks_checkout, monkeypatch):
    hardcode_benchmark_hash(monkeypatch)

    shutil.copyfile(
        DATA_PATH / "bench_runner.toml", benchmarks_checkout / "bench_runner.toml"
    )

    venv_dir = benchmarks_checkout / "venv"
    venv_python = venv_dir / "bin" / "python"

    shutil.copy(
        DATA_PATH / "benchmarks.manifest",
        benchmarks_checkout / "benchmarks.manifest",
    )

    # Now actually run the run_benchmarks.py script
    subprocess.check_call(
        [
            venv_python,
            "-m",
            "bench_runner",
            "run_benchmarks",
            "benchmark",
            sys.executable,
            "python",
            "main",
            "deepcopy",
            ",,",
            "--test_mode",
            "--run_id",
            "12345",
        ],
        cwd=benchmarks_checkout,
    )

    with open(
        benchmarks_checkout
        / "results"
        / f"bm-20220323-{platform.python_version()}-9d38120"
        / f"bm-20220323-{platform.system().lower()}-{platform.machine()}-"
        f"python-main-{platform.python_version()}-9d38120.json"
    ) as fd:
        content = json.load(fd)

    metadata = content["metadata"]
    benchmarks = content["benchmarks"]

    assert metadata["commit_id"] == "9d38120"
    assert metadata["commit_fork"] == "python"
    assert metadata["commit_branch"] == "main"
    assert metadata["commit_date"].startswith("2022-03-23T20:12:04")
    assert "commit_merge_base" not in metadata
    assert metadata["benchmark_hash"] == "215d35"
    assert (
        metadata["github_action_url"]
        == "https://github.com/faster-cpython/bench_runner/actions/runs/12345"
    )

    assert len(benchmarks) == 3
    assert all(len(benchmark["runs"]) > 1 for benchmark in benchmarks)
    assert set(bm["metadata"]["name"] for bm in benchmarks) == {
        "deepcopy",
        "deepcopy_memo",
        "deepcopy_reduce",
    }

    # Run an unknown benchmark, expect an error
    returncode = subprocess.call(
        [
            venv_python,
            "-m",
            "bench_runner",
            "run_benchmarks",
            "benchmark",
            sys.executable,
            "python",
            "main",
            "foo",
            ",,",
            "--run_id",
            "12345",
        ],
        cwd=benchmarks_checkout,
    )
    assert returncode == 1


def test_run_benchmarks_flags(benchmarks_checkout):
    shutil.copyfile(
        DATA_PATH / "bench_runner.toml", benchmarks_checkout / "bench_runner.toml"
    )

    venv_dir = benchmarks_checkout / "venv"
    venv_python = venv_dir / "bin" / "python"

    shutil.copy(
        DATA_PATH / "benchmarks.manifest",
        benchmarks_checkout / "benchmarks.manifest",
    )

    # Now actually run the run_benchmarks.py script
    subprocess.check_call(
        [
            venv_python,
            "-m",
            "bench_runner",
            "run_benchmarks",
            "benchmark",
            sys.executable,
            "python",
            "main",
            "nbody",
            "tier2,,",
            "--test_mode",
            "--run_id",
            "12345",
        ],
        cwd=benchmarks_checkout,
    )

    with open(
        benchmarks_checkout
        / "results"
        / f"bm-20220323-{platform.python_version()}-9d38120-PYTHON_UOPS"
        / f"bm-20220323-{platform.system().lower()}-{platform.machine()}-"
        f"python-main-{platform.python_version()}-9d38120.json"
    ) as fd:
        json.load(fd)
