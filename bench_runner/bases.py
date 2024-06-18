"""
Handling the list of base versions defined in bases.txt.
"""

from __future__ import annotations


import functools
from pathlib import Path


@functools.cache
def get_bases(
    bases_filepath: Path = Path("bases.txt"),
) -> list[str]:
    with bases_filepath.open() as fd:
        return list(
            line.strip() for line in fd.readlines() if line and not line.startswith("#")
        )


@functools.cache
def get_minimum_version_for_all_comparisons(bases_filepath: Path = Path("bases.txt")):
    from packaging import version as pkg_version

    bases = get_bases(bases_filepath)
    return pkg_version.parse(bases[-1].replace("+", "0")).release[0:2]
