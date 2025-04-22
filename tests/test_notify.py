import contextlib
from pathlib import Path
import textwrap


from bench_runner import gh
from bench_runner.scripts import notify


DATA_PATH = Path(__file__).parent / "data"


def test_notify(monkeypatch, capsys):
    message_sent = [None]

    def send_notification(message):
        message_sent[0] = message

    monkeypatch.setattr(gh, "send_notification", send_notification)
    monkeypatch.setenv("GITHUB_ACTOR", "test_actor")
    monkeypatch.setenv("GITHUB_REPOSITORY", "test_repo")

    with contextlib.chdir(DATA_PATH):
        notify._main(
            fork="test_fork",
            ref="test_ref",
            head="test_head",
            date="2023-10-01",
            version="3.10.4",
            flags=["JIT", "TAILCALL"],
        )

    captured = capsys.readouterr()
    assert (
        captured.out.strip()
        == "::notice ::@test_actor: [test_fork/test_ref](https://github.com/test_repo-public/tree/main/results/bm-20231001-3.10.4-test_he-JIT,TAILCALL)"
    )

    expected = textwrap.dedent(
        """
        🤖 This is the friendly benchmarking bot with some new results!

        @test_actor: [test_fork/test_ref](https://github.com/test_repo-public/tree/main/results/bm-20231001-3.10.4-test_he-JIT,TAILCALL)
        """
    ).strip()

    assert message_sent[0].strip() == expected
