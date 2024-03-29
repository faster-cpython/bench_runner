"""
Utilities to use the `gh` CLI for workflow automation.
"""

from __future__ import annotations


from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional


from . import runners


def get_machines(path: Optional[Path] = None):
    return [x.name for x in runners.get_runners(path) if x.available] + ["all"]


def _get_flags(d: Mapping[str, Any]) -> list[str]:
    flags = []
    for key, val in d.items():
        if val is None:
            continue
        if isinstance(val, bool):
            val = str(val).lower()
        else:
            val = str(val)
        flags.extend(["-f", f"{key}={val}"])
    return flags


def benchmark(
    fork: Optional[str] = None,
    ref: Optional[str] = None,
    machine: Optional[str] = None,
    benchmark_base: Optional[bool] = None,
    tier2: Optional[bool] = None,
    _runner_path: Optional[Path] = None,
) -> None:
    if not (fork is None or isinstance(fork, str)):
        raise TypeError(f"fork must be a str, got {type(fork)}")

    if not (ref is None or isinstance(ref, str)):
        raise TypeError(f"ref must be a str, got {type(ref)}")

    machines = get_machines(_runner_path)
    if not (machine is None or machine in machines):
        raise ValueError(f"machine must be one of {machines}")

    if not (benchmark_base is None or isinstance(benchmark_base, bool)):
        raise TypeError(f"benchmark_base must be bool, got {type(benchmark_base)}")

    if tier2 is None:
        tier2 = False

    flags = _get_flags(
        {
            "fork": fork,
            "ref": ref,
            "machine": machine,
            "benchmark_base": benchmark_base,
            "tier2": str(tier2).lower(),
        }
    )

    subprocess.check_call(
        ["gh", "workflow", "run", "benchmark.yml", *flags],
    )


def send_notification(body):
    print("Sending Github notification:")
    print("---")
    print(body)
    print("---")
    subprocess.check_call(["gh", "issue", "comment", "182", "--body", body])
