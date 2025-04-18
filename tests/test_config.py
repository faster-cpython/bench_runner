from pathlib import Path
import socket


from bench_runner import config
from bench_runner import runners


DATA_PATH = Path(__file__).parent / "data"


def test_get_runner_for_hostname(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "pyperf")

    runner = config.get_config_for_current_runner(DATA_PATH / "bench_runner.toml")

    assert runner["os"] == "linux"
    assert runner["arch"] == "x86_64"
    assert runner["hostname"] == "pyperf"
