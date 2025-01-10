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
    Flag("tier2", "PYTHON_UOPS", "tier 2 interpreter", "T2"),
    Flag("jit", "JIT", "JIT", "JIT"),
    Flag("nogil", "NOGIL", "free threading", "NOGIL"),
    Flag("clang", "CLANG", "build with latest clang", "CLANG"),
]


FLAG_MAPPING = {flag.gha_variable: flag.name for flag in FLAGS}


def parse_flags(flag_str: str | None) -> list[str]:
    if flag_str is None:
        return []

    flag_mapping = FLAG_MAPPING
    flags = flag_str.split(",")

    internal_flags = []
    for flag in flags:
        stripped_flag = flag.strip()
        if stripped_flag:
            if stripped_flag not in flag_mapping:
                raise ValueError(f"Invalid flag {stripped_flag}")
            internal_flags.append(flag_mapping[stripped_flag])

    return internal_flags


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
