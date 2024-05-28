import contextlib
from pathlib import Path
import shutil
import subprocess
import textwrap


import pytest


from bench_runner.scripts import get_merge_base


DATA_PATH = Path(__file__).parent / "data"


@pytest.fixture
def checkout(request):
    root = request.config.cache.mkdir("get-merge-base-checkout")

    if not (root / "cpython").is_dir():
        subprocess.check_call(
            [
                "git",
                "clone",
                "https://github.com/mdboom/cpython",
                "--branch",
                "fix-pystats",
                "--depth",
                "50",
            ],
            cwd=root,
        )

    return root


def test_get_merge_base(tmp_path, capsys, checkout, monkeypatch):
    monkeypatch.chdir(DATA_PATH)

    shutil.copytree(checkout / "cpython", tmp_path / "cpython")

    get_merge_base._main(True, "linux-x86_64-linux", False, [], tmp_path / "cpython")

    captured = capsys.readouterr()

    assert (
        captured.out.strip()
        == textwrap.dedent(
            """
    head=efa0bb6
    date=2022-12-12T08:45:23-05:00
    version=3.12.0a3+
    ref=158b8a07212cea6066afe8bb91f1cd542d922dba
    need_to_run=true
    """
        ).strip()
    )


def test_hard_coded(tmp_path, capsys, checkout, monkeypatch):
    monkeypatch.chdir(DATA_PATH)

    shutil.copytree(checkout / "cpython", tmp_path / "cpython")

    with contextlib.chdir(tmp_path):
        get_merge_base._main(False, "linux-x86_64-linux", False, [])

    captured = capsys.readouterr()

    assert "need_to_run=false" in captured.out
