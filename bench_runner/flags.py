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
    Flag("tailcall", "TAILCALL", "tail-calling interpreter", "TAILCALL"),
]


FLAG_MAPPING = {flag.gha_variable: flag.name for flag in FLAGS}


def parse_flags(flag_str: str | None) -> list[str]:
    if flag_str is None:
        return []
    flags = [flag.strip() for flag in flag_str.split(",") if flag.strip() != ""]
    internal_flags = []
    for flag in flags:
        if flag not in FLAG_MAPPING:
            raise ValueError(f"Invalid flag {flag}")
        internal_flags.append(FLAG_MAPPING[flag])
    return internal_flags


def flags_to_gha_variables(flags: list[str]) -> dict[str, str]:
    return {
        k: "true" if v else "false"
        for k, v in flags_to_gha_variables_yml(flags).items()
    }


def flags_to_gha_variables_yml(flags: list[str]) -> dict[str, bool]:
    output = {}
    for flag_descr in FLAGS:
        output[flag_descr.gha_variable] = flag_descr.name in flags
    return output


def flags_to_human(flags: list[str]) -> Iterable[str]:
    for flag in flags:
        for flag_descr in FLAGS:
            if flag_descr.name == flag:
                yield flag_descr.short_name
                break


def normalize_flags(flags: list[str]) -> list[str]:
    result = sorted(set(flags))
    for flag in result:
        if flag not in FLAG_MAPPING.values():
            raise ValueError(f"Invalid flag {flag}")
    return result
