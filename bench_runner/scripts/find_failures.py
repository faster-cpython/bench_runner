from __future__ import annotations


from collections import defaultdict
import json
import re
import subprocess
import sys
from typing import Any, Iterator, Mapping, TextIO, TypeAlias


from bench_runner import table


Failures: TypeAlias = Mapping[str, Mapping[str, Mapping[str, tuple[str, list[str]]]]]


def iter_lines(log: str) -> Iterator[tuple[str, str, str]]:
    for line in log.splitlines():
        parts = line.split("\t", maxsplit=2)
        if len(parts) > 2:
            subparts = parts[2].split(" ", maxsplit=1)
            if len(subparts) > 1:
                config_machine = parts[0]
                if " / " in config_machine:
                    config, machine = config_machine.split(" / ")
                    if "benchmark-" in machine:
                        machine = machine.split("-")[1]
                    config = config.split("-")[-1]
                    if config == "weekly":
                        config = "default"
                    yield machine, config, subparts[1]


def iter_configs(configs: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    for default in ("default", "pystats"):
        if default in configs:
            yield default, configs[default]
            break
    else:
        yield from configs.items()


def get_last_weekly_run_id() -> str:
    output = subprocess.check_output(
        ["gh", "run", "list", "--workflow", "_weekly.yml", "--json", "databaseId"]
    )
    content = json.loads(output)
    return content[0]["databaseId"]


def get_log(run_id: str) -> str:
    return subprocess.check_output(["gh", "run", "view", "--log", str(run_id)]).decode(
        "utf-8"
    )


def parse_log(content: str) -> Failures:
    failures = defaultdict(lambda: defaultdict(dict))

    iter = iter_lines(content)

    collected_lines = []
    current_benchmark_build = None
    current_benchmark_run = None
    benchmark_run_pattern = re.compile(r"\[.+\] (.+)\.\.\.")
    benchmark_fail_pattern = re.compile(r"ERROR: Benchmark (.+) failed: Benchmark died")
    venv_creation_pattern = re.compile(r"\(.+\) creating venv for benchmark \((.+)\)")

    for machine, config, line in iter:
        if match := venv_creation_pattern.match(line):
            current_benchmark_build = match.group(1)
            collected_lines = []
        elif line == "(benchmark will be skipped)":
            assert current_benchmark_build
            failures[current_benchmark_build][machine][config] = (
                "build",
                collected_lines,
            )
        elif match := benchmark_run_pattern.match(line):
            current_benchmark_run = match.group(1)
            collected_lines = []
        elif match := benchmark_fail_pattern.match(line):
            assert current_benchmark_run
            failures[current_benchmark_run][machine][config] = ("run", collected_lines)
        else:
            collected_lines.append(line)

    return failures


def write_output(fd: TextIO, failures: Failures) -> None:
    machines = set()
    for value in failures.values():
        machines.update(value.keys())
    machines = sorted(list(machines))

    rows = []
    for bm, entries in sorted(list(failures.items())):
        col = [bm]
        for machine in machines:
            if configs := entries.get(machine):
                col.append(
                    " ".join(
                        f"[[{config} {failure_type}]](#{bm}-on-{machine}-{config})"
                        for config, (failure_type, _) in iter_configs(configs)
                    )
                )
            else:
                col.append("")
        rows.append(col)
    table.output_table(fd, ["benchmark"] + machines, rows)

    for bm, entries in sorted(list(failures.items())):
        for machine, configs in entries.items():
            for config, (_, loglines) in iter_configs(configs):
                fd.write(f"## {bm} on {machine} {config}\n\n")
                table.write_details(
                    fd,
                    f"Log for {bm} on {machine} {config}",
                    ["```"] + loglines[-50:] + ["```"],
                )


def _main():
    last_run_id = get_last_weekly_run_id()
    failures = parse_log(get_log(last_run_id))
    write_output(sys.stdout, failures)


def main():
    _main()


if __name__ == "__main__":
    main()
