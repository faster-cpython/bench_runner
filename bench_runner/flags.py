from __future__ import annotations


import dataclasses
from typing import Iterable


@dataclasses.dataclass
class Flag:
    gha_variable: str
    name: str
    description: str
    short_name: str


FLAGS = [
    Flag("tier2", "PYTHON_UOPS", "Enable the Tier 2 interpreter", "T2"),
    Flag("jit", "JIT", "Enable the JIT", "JIT"),
    Flag("nogil", "NOGIL", "Enable free threading", "NOGIL"),
]


VALID_FLAGS = {flag.gha_variable for flag in FLAGS}


def parse_flags(flag_str: str) -> list[str]:
    flags = [flag.strip() for flag in flag_str.split(",") if flag.strip() != ""]
    if any(flag not in VALID_FLAGS for flag in flags):
        raise ValueError(f"Invalid flags {flag_str:r}")
    return flags


def flags_to_gha_variables(flags: list[str]) -> dict[str, str]:
    output = {}
    for flag_descr in FLAGS:
        if flag_descr.name in flags:
            output[flag_descr.gha_variable] = "true"
        else:
            output[flag_descr.gha_variable] = "false"
    return output


def flags_to_human(flags: list[str]) -> Iterable[str]:
    for flag in flags:
        for flag_descr in FLAGS:
            if flag_descr.name == flag:
                yield flag_descr.short_name
                break
