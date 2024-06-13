"""
Handling the list of base versions defined in bases.txt.
"""

from __future__ import annotations


from pathlib import Path


def get_bases(
    bases_filepath: Path = Path("bases.txt"),
) -> list[str]:
    with bases_filepath.open() as fd:
        return list(
            line.strip() for line in fd.readlines() if line and not line.startswith("#")
        )
