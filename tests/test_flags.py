import pytest


from bench_runner import flags


def test_parse_flags():
    assert flags.parse_flags("tier2,jit") == ["PYTHON_UOPS", "JIT"]
    assert flags.parse_flags("jit,tier2") == ["JIT", "PYTHON_UOPS"]
    assert flags.parse_flags("jit,tier2,") == ["JIT", "PYTHON_UOPS"]
    assert flags.parse_flags(",jit,tier2,") == ["JIT", "PYTHON_UOPS"]
    assert flags.parse_flags(",") == []
    assert flags.parse_flags("") == []

    with pytest.raises(ValueError):
        flags.parse_flags("tier2,jit,foo")


def test_flags_to_human():
    assert list(flags.flags_to_human(["PYTHON_UOPS", "JIT"])) == ["T2", "JIT"]
