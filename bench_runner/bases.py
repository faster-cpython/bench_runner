"""
Handling the list of base versions defined in bench_runner.toml.
"""

from __future__ import annotations


import functools


from . import config


def get_bases() -> list[str]:
    return config.get_bench_runner_config().get("bases", {}).get("versions", [])


@functools.cache
def get_minimum_version_for_all_comparisons():
    from packaging import version as pkg_version

    bases = get_bases()
    try:
        return pkg_version.parse(bases[-1].replace("+", "0")).release[0:2]
    except pkg_version.InvalidVersion:
        return (0, 0)
