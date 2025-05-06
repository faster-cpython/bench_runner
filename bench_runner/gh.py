"""
Utilities to use the `gh` CLI for workflow automation.
"""

from __future__ import annotations


import subprocess
from typing import Any, Mapping


from . import config
from . import flags as mflags


def get_machines():
    cfg = config.get_config()
    return [x.name for x in cfg.runners.values() if x.available] + ["all"]


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
    fork: str | None = None,
    ref: str | None = None,
    machine: str | None = None,
    benchmark_base: bool | None = None,
    flags: list[str] | None = None,
) -> None:
    if not (fork is None or isinstance(fork, str)):
        raise TypeError(f"fork must be a str, got {type(fork)}")

    if not (ref is None or isinstance(ref, str)):
        raise TypeError(f"ref must be a str, got {type(ref)}")

    machines = get_machines()
    if not (machine is None or machine in machines):
        raise ValueError(f"machine must be one of {machines}")

    if not (benchmark_base is None or isinstance(benchmark_base, bool)):
        raise TypeError(f"benchmark_base must be bool, got {type(benchmark_base)}")

    if flags is None:
        flags = []

    args = {
        "fork": fork,
        "ref": ref,
        "machine": machine,
        "benchmark_base": benchmark_base,
    }

    args.update(mflags.flags_to_gha_variables(flags))

    cli_flags = _get_flags(args)

    subprocess.check_call(
        ["gh", "workflow", "run", "benchmark.yml", *cli_flags],
    )


def send_notification(body):
    cfg = config.get_config()
    notification_issue = cfg.notify.notification_issue

    if notification_issue == 0:
        print("Not sending Github notification.")
        return

    print("Sending Github notification:")
    print("---")
    print(body)
    print("---")
    subprocess.check_call(
        ["gh", "issue", "comment", str(notification_issue), "--body", body]
    )
