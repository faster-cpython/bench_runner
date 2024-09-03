from pathlib import Path


from bench_runner import bases as mod_bases


DATA_PATH = Path(__file__).parent / "data"


def test_get_bases(monkeypatch):
    monkeypatch.chdir(DATA_PATH)
    bases = mod_bases.get_bases()
    assert bases == ["base2", "base4"]
