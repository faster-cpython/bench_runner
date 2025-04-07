import json
import pathlib
import sys
import tempfile

import pytest

from bench_runner.scripts import synthesize_loops_file

DATA_PATH = pathlib.Path(__file__).parent / "data"


def run_synthesize(
    output: pathlib.Path,
    datadir: pathlib.Path,
    *,
    update: bool = False,
    overwrite: bool = False,
    merger: str = "max",
):
    files = datadir.glob("results/**/*.json")
    synthesize_loops_file._main(
        loops_file=output,
        update=update,
        overwrite=overwrite,
        merger=merger,
        results=files,
    )


def check_loops(output: pathlib.Path):
    with output.open() as f:
        data = json.load(f)
    assert "benchmarks" in data
    assert "metadata" in data
    seen = set()
    for bm in data["benchmarks"]:
        assert "metadata" in bm
        assert "loops" in bm["metadata"]
        assert isinstance(bm["metadata"]["loops"], int)
        assert "name" in bm["metadata"]
        assert (name := bm["metadata"]["name"]) not in seen
        assert isinstance(name, str)
        seen.add(name)
    data["benchmarks"].sort(key=lambda item: item["metadata"]["name"])
    return data


def set_loops(output, value):
    with output.open() as f:
        data = json.load(f)
    for bm in data["benchmarks"]:
        bm["metadata"]["loops"] = value
    with output.open("w") as f:
        json.dump(data, f, sort_keys=True, indent=4)


def test_synthesize():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = pathlib.Path(tmpdir) / "loops.json"
        run_synthesize(output, DATA_PATH)
        expected_data = check_loops(output)

        with pytest.raises(FileExistsError):
            run_synthesize(output, DATA_PATH)

        run_synthesize(output, DATA_PATH, update=True)
        assert expected_data == check_loops(output)

        set_loops(output, 0)
        run_synthesize(output, DATA_PATH, update=True)
        assert expected_data == check_loops(output)

        set_loops(output, sys.maxsize)
        run_synthesize(output, DATA_PATH, overwrite=True)
        assert expected_data == check_loops(output)

        run_synthesize(output, DATA_PATH, overwrite=True, merger="min")
        expected_data = check_loops(output)
        set_loops(output, sys.maxsize)
        run_synthesize(output, DATA_PATH, update=True, merger="min")
        assert expected_data == check_loops(output)
