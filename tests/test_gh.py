from pathlib import Path
import subprocess


import pytest


from bench_runner import gh


DATA_PATH = Path(__file__).parent / "data"


def test_benchmark_arguments():
    runner_path = DATA_PATH / "runners.ini"

    with pytest.raises(TypeError):
        gh.benchmark(fork=1)

    with pytest.raises(TypeError):
        gh.benchmark(ref=1)

    with pytest.raises(ValueError):
        gh.benchmark(machine="", _runner_path=runner_path)

    with pytest.raises(ValueError):
        gh.benchmark(machine="linux-x86_64", _runner_path=runner_path)

    with pytest.raises(TypeError):
        gh.benchmark(benchmark_base=1, _runner_path=runner_path)


def test_benchmark_cmdline(monkeypatch):
    runner_path = DATA_PATH / "runners.ini"

    args_out = None

    def get_args(args, **kwargs):
        nonlocal args_out
        args_out = args

    monkeypatch.setattr(subprocess, "check_call", get_args)

    gh.benchmark(fork="myfork", benchmark_base=True, _runner_path=runner_path)

    assert args_out == [
        "gh",
        "workflow",
        "run",
        "benchmark.yml",
        "-f",
        "fork=myfork",
        "-f",
        "benchmark_base=true",
        "-f",
        "tier2=false",
        "-f",
        "jit=false",
        "-f",
        "nogil=false",
    ]
