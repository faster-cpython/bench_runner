import contextlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


import pytest


from bench_runner import git
from bench_runner.scripts import generate_results
from bench_runner.scripts import run_benchmarks
from bench_runner.scripts import should_run
from bench_runner import util


DATA_PATH = Path(__file__).parent / "data"


def _copy_repo(tmp_path):
    repo_path = tmp_path / "repo"
    shutil.copytree(DATA_PATH, repo_path)
    return repo_path


def dont_get_git_merge_base(monkeypatch):
    def dummy(*args, **kwargs):
        return None

    monkeypatch.setattr(git, "get_git_merge_base", dummy)


def test_update_metadata(benchmarks_checkout, monkeypatch):
    dont_get_git_merge_base(monkeypatch)

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


def test_run_benchmarks(benchmarks_checkout):
    shutil.copyfile(DATA_PATH / "runners.ini", benchmarks_checkout / "runners.ini")

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
            run_benchmarks.__file__,
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
            run_benchmarks.__file__,
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


def test_should_run_exists_noforce(benchmarks_checkout, capsys, monkeypatch):
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)

    should_run._main(
        False,
        "python",
        "main",
        "linux-x86_64-linux",
        False,
        ",,",
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    captured = capsys.readouterr()
    assert captured.out.strip() == "should_run=false"
    assert (repo / "results" / "bm-20220323-3.10.4-9d38120").is_dir()


def test_should_run_diff_machine_noforce(benchmarks_checkout, capsys, monkeypatch):
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)

    should_run._main(
        False,
        "python",
        "main",
        "darwin-x86_64-darwin",
        False,
        ",,",
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    captured = capsys.readouterr()
    assert captured.out.strip() == "should_run=true"
    assert len(list((repo / "results" / "bm-20220323-3.10.4-9d38120").iterdir())) == 1


def test_should_run_all_noforce(benchmarks_checkout, capsys, monkeypatch):
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)

    should_run._main(
        False,
        "python",
        "main",
        "all",
        False,
        ",,",
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    captured = capsys.readouterr()
    assert captured.out.strip() == "should_run=true"
    assert len(list((repo / "results" / "bm-20220323-3.10.4-9d38120").iterdir())) == 1


def test_should_run_noexists_noforce(benchmarks_checkout, capsys, monkeypatch):
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)
    shutil.rmtree(repo / "results" / "bm-20220323-3.10.4-9d38120")

    should_run._main(
        False,
        "python",
        "main",
        "linux-x86_64-linux",
        False,
        ",,",
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    captured = capsys.readouterr()
    assert captured.out.strip() == "should_run=true"
    assert not (repo / "results" / "bm-20220323-3.10.4-9d38120").is_dir()


def test_should_run_exists_force(benchmarks_checkout, capsys, monkeypatch):
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)

    removed_paths = []

    def remove(repo, path):
        nonlocal removed_paths
        removed_paths.append(path)
        (repo / path).unlink()

    monkeypatch.setattr(git, "remove", remove)

    with contextlib.chdir(repo):
        generate_results._main(repo, force=False, bases=["3.11.0b3"])
        should_run._main(
            True,
            "python",
            "main",
            "linux-x86_64-linux",
            False,
            ",,",
            benchmarks_checkout / "cpython",
            repo / "results",
        )

    captured = capsys.readouterr()
    assert captured.out.splitlines()[-1].strip() == "should_run=true"
    assert (repo / "results" / "bm-20220323-3.10.4-9d38120").is_dir()
    assert set(x.name for x in removed_paths) == {
        "bm-20220323-linux-x86_64-python-main-3.10.4-9d38120-vs-3.11.0b3.svg",
        "README.md",
        "bm-20220323-linux-x86_64-python-main-3.10.4-9d38120-vs-3.11.0b3.md",
    }


def test_should_run_noexists_force(benchmarks_checkout, capsys, monkeypatch):
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)
    shutil.rmtree(repo / "results" / "bm-20220323-3.10.4-9d38120")

    should_run._main(
        True,
        "python",
        "main",
        "linux-x86_64-linux",
        False,
        ",,",
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    captured = capsys.readouterr()
    assert captured.out.strip() == "should_run=true"
    assert not (repo / "results" / "bm-20220323-3.10.4-9d38120").is_dir()


def test_should_run_checkout_failed(tmp_path, capsys, monkeypatch):
    repo = _copy_repo(tmp_path)
    monkeypatch.chdir(repo)
    cpython_path = tmp_path / "cpython"
    cpython_path.mkdir()
    subprocess.check_call(["git", "init"], cwd=cpython_path)

    with pytest.raises(SystemExit):
        should_run._main(
            True,
            "python",
            "main",
            "linux-x86_64-linux",
            False,
            ",,",
            cpython_path,
            repo / "results",
        )

    captured = capsys.readouterr()
    assert "The checkout of cpython failed" in captured.err
    assert "You specified fork 'python' and ref 'main'" in captured.err


def test_run_benchmarks_flags(benchmarks_checkout):
    shutil.copyfile(DATA_PATH / "runners.ini", benchmarks_checkout / "runners.ini")

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
            run_benchmarks.__file__,
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


def test_get_benchmark_hash():
    assert util.get_benchmark_hash() == "215d35"
