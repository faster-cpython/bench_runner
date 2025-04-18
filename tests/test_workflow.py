import contextlib
from pathlib import Path
import shutil
import subprocess
import sys


import pytest


from bench_runner import benchmark_definitions
from bench_runner import git
from bench_runner.scripts import generate_results
from bench_runner.scripts import workflow


DATA_PATH = Path(__file__).parent / "data"


def _copy_repo(tmp_path):
    repo_path = tmp_path / "repo"
    shutil.copytree(DATA_PATH, repo_path)
    return repo_path


def hardcode_benchmark_hash(monkeypatch):
    def dummy(*args, **kwargs):
        return "215d35"

    monkeypatch.setattr(benchmark_definitions, "get_benchmark_hash", dummy)


def test_run_in_venv(tmpdir):
    venv_dir = tmpdir / "venv"

    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    workflow.run_in_venv(venv_dir, "pip", ["install", "rich"])
    workflow.run_in_venv(venv_dir, "rich", [])


def test_should_run_exists_noforce(benchmarks_checkout, monkeypatch):
    hardcode_benchmark_hash(monkeypatch)
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)

    result = workflow.should_run(
        False,
        "python",
        "main",
        "linux-x86_64-linux",
        False,
        [],
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    assert result is False
    assert (repo / "results" / "bm-20220323-3.10.4-9d38120").is_dir()


def test_should_run_diff_machine_noforce(benchmarks_checkout, monkeypatch):
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)

    result = workflow.should_run(
        False,
        "python",
        "main",
        "darwin-x86_64-darwin",
        False,
        [],
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    assert result is True
    assert len(list((repo / "results" / "bm-20220323-3.10.4-9d38120").iterdir())) == 1


def test_should_run_all_noforce(benchmarks_checkout, monkeypatch):
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)

    result = workflow.should_run(
        False,
        "python",
        "main",
        "all",
        False,
        [],
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    assert result is True
    assert len(list((repo / "results" / "bm-20220323-3.10.4-9d38120").iterdir())) == 1


def test_should_run_noexists_noforce(benchmarks_checkout, monkeypatch):
    hardcode_benchmark_hash(monkeypatch)
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)
    shutil.rmtree(repo / "results" / "bm-20220323-3.10.4-9d38120")

    result = workflow.should_run(
        False,
        "python",
        "main",
        "linux-x86_64-linux",
        False,
        [],
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    assert result is True
    assert not (repo / "results" / "bm-20220323-3.10.4-9d38120").is_dir()


def test_should_run_exists_force(benchmarks_checkout, monkeypatch):
    hardcode_benchmark_hash(monkeypatch)

    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)

    removed_paths = []

    def remove(repo, path):
        removed_paths.append(path)
        (repo / path).unlink()

    monkeypatch.setattr(git, "remove", remove)

    generate_results._main(repo, force=False, bases=["3.11.0b3"])
    result = workflow.should_run(
        True,
        "python",
        "main",
        "linux-x86_64-linux",
        False,
        [],
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    assert result is True
    assert (repo / "results" / "bm-20220323-3.10.4-9d38120").is_dir()
    assert set(x.name for x in removed_paths) == {
        "bm-20220323-linux-x86_64-python-main-3.10.4-9d38120-vs-3.11.0b3.svg",
        "README.md",
        "bm-20220323-linux-x86_64-python-main-3.10.4-9d38120-vs-3.11.0b3.md",
    }


def test_should_run_noexists_force(benchmarks_checkout, monkeypatch):
    hardcode_benchmark_hash(monkeypatch)
    repo = _copy_repo(benchmarks_checkout)
    monkeypatch.chdir(repo)
    shutil.rmtree(repo / "results" / "bm-20220323-3.10.4-9d38120")

    result = workflow.should_run(
        True,
        "python",
        "main",
        "linux-x86_64-linux",
        False,
        [],
        benchmarks_checkout / "cpython",
        repo / "results",
    )

    assert result is True
    assert not (repo / "results" / "bm-20220323-3.10.4-9d38120").is_dir()


def test_should_run_checkout_failed(tmp_path, capsys, monkeypatch):
    repo = _copy_repo(tmp_path)
    monkeypatch.chdir(repo)
    cpython_path = tmp_path / "cpython"
    cpython_path.mkdir()
    subprocess.check_call(["git", "init"], cwd=cpython_path)

    with pytest.raises(SystemExit):
        workflow.should_run(
            True,
            "python",
            "main",
            "linux-x86_64-linux",
            False,
            [],
            cpython_path,
            repo / "results",
        )

    captured = capsys.readouterr()
    assert "The checkout of cpython failed" in captured.err
    assert "You specified fork 'python' and ref 'main'" in captured.err


@pytest.mark.long_running
def test_whole_workflow(tmpdir):
    """
    Tests the whole workflow from a clean benchmarking repo.
    """
    repo = tmpdir / "repo"
    venv_dir = repo / "outer_venv"
    bench_runner_checkout = DATA_PATH.parents[1]
    if sys.platform.startswith("win"):
        binary = venv_dir / "Scripts" / "python.exe"
    else:
        binary = venv_dir / "bin" / "python"

    repo.mkdir()

    with contextlib.chdir(repo):
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        subprocess.check_call(
            [
                str(binary),
                "-m",
                "pip",
                "install",
                str(bench_runner_checkout),
            ]
        )
        subprocess.check_call([str(binary), "-m", "bench_runner", "install"])
        # install --check should never fail immediately after install
        subprocess.check_call(
            [
                str(binary),
                "-m",
                "bench_runner",
                "install",
                "--check",
            ]
        )
        with open("requirements.txt", "w") as fd:
            fd.write(f"{str(bench_runner_checkout)}\n")
        subprocess.check_call(
            [
                str(binary),
                "workflow_bootstrap.py",
                "python",
                "main",
                "linux-x86_64-linux",
                "deltablue",
                ",,,",
                "--_fast",
            ]
        )
