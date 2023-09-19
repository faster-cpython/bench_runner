from pathlib import Path
import shutil
import textwrap


from bench_runner.scripts import compare

DATA_PATH = Path(__file__).parent / "data"


def _copy_repo(tmp_path):
    repo_path = tmp_path / "repo"
    shutil.copytree(DATA_PATH, repo_path)
    return repo_path


def test_compare_1_to_n(tmp_path, monkeypatch):
    repo_path = _copy_repo(tmp_path)
    monkeypatch.chdir(repo_path)
    output_path = tmp_path / "output"

    hashes = ["9d38120", "eb0004c", "b0e1f9c"]

    compare.main(hashes, output_path, "1:n")

    files = list(output_path.iterdir())
    assert len(files) == 5
    expected_files = {"README.md"}
    for hash in hashes[1:]:
        for suffix in ["md", "png"]:
            expected_files.add(f"{hash}-vs-{hashes[0]}.{suffix}")
    assert set(x.name for x in files) == expected_files

    content = (output_path / "README.md").read_text()
    expected = textwrap.dedent(
        """
        # linux

        | commit | change |
        | -- | -- |
        | eb0004c (eb0004c) | 1.57x faster (94% rel.) [table](eb0004c-vs-9d38120.md) [plot](eb0004c-vs-9d38120.png) |
        | b0e1f9c (b0e1f9c) | 1.70x faster \* (94% rel.) [table](b0e1f9c-vs-9d38120.md) [plot](b0e1f9c-vs-9d38120.png) |
        """
    ).strip()
    assert expected in content


def test_compare_n_to_n(tmp_path, monkeypatch):
    repo_path = _copy_repo(tmp_path)
    monkeypatch.chdir(repo_path)
    output_path = tmp_path / "output"

    hashes = ["9d38120", "eb0004c", "b0e1f9c"]

    compare.main(hashes, output_path, "n:n")

    files = list(output_path.iterdir())
    assert len(files) == 13
    expected_files = {"README.md"}
    for hash1 in hashes:
        for hash2 in hashes:
            if hash1 == hash2:
                continue
            for suffix in ["md", "png"]:
                expected_files.add(f"{hash2}-vs-{hash1}.{suffix}")
    assert set(x.name for x in files) == expected_files

    content = (output_path / "README.md").read_text()
    expected = textwrap.dedent(
        """
        # linux

        |  | 9d38120 (9d38120) | eb0004c (eb0004c) | b0e1f9c (b0e1f9c) |
        | -- | -- | -- | -- |
        | 9d38120 | | 1.57x faster (94% rel.) [table](eb0004c-vs-9d38120.md) [plot](eb0004c-vs-9d38120.png) | 1.70x faster \* (94% rel.) [table](b0e1f9c-vs-9d38120.md) [plot](b0e1f9c-vs-9d38120.png) |
        | eb0004c | 1.57x slower (94% rel.) [table](9d38120-vs-eb0004c.md) [plot](9d38120-vs-eb0004c.png) | | 1.08x faster \* (94% rel.) [table](b0e1f9c-vs-eb0004c.md) [plot](b0e1f9c-vs-eb0004c.png) |
        | b0e1f9c | 1.70x slower \* (94% rel.) [table](9d38120-vs-b0e1f9c.md) [plot](9d38120-vs-b0e1f9c.png) | 1.08x slower \* (94% rel.) [table](eb0004c-vs-b0e1f9c.md) [plot](eb0004c-vs-b0e1f9c.png) | |
        """
    ).strip()
    assert expected in content
