"""
Regenerates some Github Actions workflow files from templates.
"""

import argparse
import copy
import functools
import io
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, TextIO


import rich
import rich_argparse
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString


from bench_runner import config
from bench_runner import flags
from bench_runner import runners
from bench_runner.util import PathLike


ROOT_PATH = Path()
TEMPLATE_PATH = Path(__file__).parents[1] / "templates"
WORKFLOW_PATH = Path() / ".github" / "workflows"


def fail_check(dst: PathLike):
    rich.print(
        f"[red]{Path(dst).relative_to(ROOT_PATH)} needs to be regenerated.[/red]",
        file=sys.stderr,
    )
    rich.print(
        "Run `python -m bench_runner install` and commit the result.",
        file=sys.stderr,
    )
    sys.exit(1)


def write_and_check(dst: PathLike, writer: Callable[[TextIO], None], check: bool):
    """
    Call `writer` with a file descriptor to write the contents to `dst`.

    If `check` is True, raise SystemExit if the file would change. This is used
    in CI to confirm that the file was regenerated after changes to the source
    file.
    """
    dst = Path(dst)

    if check:
        if not dst.is_file():
            fail_check(dst)

        fd = io.StringIO()
        orig_contents = dst.read_text()
        writer(fd)
        new_contents = fd.getvalue()
        if orig_contents != new_contents:
            fail_check(dst)
    else:
        with dst.open("w") as fd:
            writer(fd)


def write_yaml(dst: PathLike, contents: Any, check: bool):
    """
    Write `contents` to `dst` as YAML.

    If `check` is True, raise SystemExit if the file would change. This is used
    in CI to confirm that the file was regenerated after changes to the source
    file.
    """

    def do_write(fd):
        fd.write("# Generated file: !!! DO NOT EDIT !!!\n")
        fd.write("---\n")
        yaml = YAML()
        yaml.dump(contents, fd)

    return write_and_check(dst, do_write, check)


def load_yaml(src: PathLike) -> Any:
    """
    Load YAML from `src`.
    """
    with Path(src).open() as fd:
        yaml = YAML()
        return yaml.load(fd)


def write_python(dst: PathLike, contents: str, check: bool):
    """
    Write a string of Python code to a file, adding a header about it being generated.

    If `check` is True, raise SystemExit if the file would change. This is used
    in CI to confirm that the file was regenerated after changes to the source
    file.
    """

    def do_write(fd):
        fd.write("# Generated file: !!! DO NOT EDIT !!!\n\n")
        fd.write(contents)

    return write_and_check(dst, do_write, check)


def add_flag_variables(dst: dict[str, Any]) -> None:
    for flag in flags.FLAGS:
        dst[flag.gha_variable] = {
            "description": flag.description,
            "type": "boolean",
            "default": False,
        }


@functools.cache
def flag_env():
    return ",".join(
        f"${{{{ inputs.{flag.gha_variable} == true && '{flag.gha_variable}' || '' }}}}"
        for flag in flags.FLAGS
    )


def add_flag_env(jobs: dict[str, Any]):
    flag_value = flag_env()  # Compute flag_env once and reuse the result
    for job in jobs.values():
        if "steps" in job:
            job.setdefault("env", {})
            job["env"]["flags"] = flag_value
            for step in job["steps"]:
                if "run" in step:
                    step["run"] = step["run"].replace("${{ env.flags }}", flag_value)


def generate__benchmark(src: Any) -> Any:
    """
    Generates _benchmark.yml from _benchmark.src.yml.

    For each runner machine, inserts the platform-specific set of steps for
    that machine.

    Inserts the list of available machines to the drop-down presented to the
    user.
    """
    cfg = config.get_config()
    available_runners = [r for r in cfg.runners.values() if r.available]
    runner_choices = [*[x.name for x in available_runners], "all"]

    dst = copy.deepcopy(src)

    dst["jobs"] = {}
    for runner in available_runners:
        runner_template = copy.deepcopy(src["jobs"][f"benchmark-{runner.os}"])

        # Set environment variables for the runner
        if runner.os == "windows":
            # Powershell syntax
            github_env = "$env:GITHUB_ENV"
        else:
            # sh syntax
            github_env = "$GITHUB_ENV"
        vars = copy.copy(runner.env)
        vars["BENCHMARK_MACHINE_NICKNAME"] = runner.nickname
        setup_environment = {
            "name": "Setup environment",
            "run": LiteralScalarString(
                "\n".join(
                    f'echo "{key}={val}" >> {github_env}' for key, val in vars.items()
                )
            ),
        }
        runner_template["steps"].insert(0, setup_environment)

        runner_template["runs-on"].append(runner.github_runner_name)

        machine_clauses = [
            f"inputs.machine == '{runner.name}'",
            "inputs.machine == '__really_all'",
        ]
        if runner.include_in_all:
            machine_clauses.append("inputs.machine == 'all'")
        runner_template["if"] = f"${{{{ ({' || '.join(machine_clauses)}) }}}}"

        dst["jobs"][f"benchmark-{runner.name}"] = runner_template

    add_flag_env(dst["jobs"])

    dst["on"]["workflow_dispatch"]["inputs"]["machine"]["options"] = runner_choices

    add_flag_variables(dst["on"]["workflow_dispatch"]["inputs"])
    add_flag_variables(dst["on"]["workflow_call"]["inputs"])

    return dst


def generate_benchmark(dst: Any) -> Any:
    """
    Generates benchmark.yml from benchmark.src.yml.

    Inserts the list of available machines to the drop-down presented to the
    user.
    """
    cfg = config.get_config()
    available_runners = [r for r in cfg.runners.values() if r.available]
    runner_choices = [*[x.name for x in available_runners], "all", "__really_all"]

    dst["on"]["workflow_dispatch"]["inputs"]["machine"]["options"] = runner_choices

    add_flag_variables(dst["on"]["workflow_dispatch"]["inputs"])
    add_flag_env(dst["jobs"])

    # Set all of the flag inputs that are delegated to the reusable workflows
    for name, job in dst["jobs"].items():
        if name == "generate":
            continue
        if "with" in job:
            for flag in flags.FLAGS:
                job["with"][
                    flag.gha_variable
                ] = f"${{{{ inputs.{flag.gha_variable} }}}}"

    # Include all of the flags in the human-readable workflow "run name"
    dst["run-name"] += " " + " ".join(
        f"${{{{ inputs.{flag.gha_variable} == true && '{flag.short_name}' || '' }}}}"
        for flag in flags.FLAGS
    )

    if cfg.publish_mirror.skip:
        del dst["jobs"]["publish"]
        dst["jobs"]["notify"]["needs"] = ["generate", "determine_base"]

    return dst


def generate__pystats(dst: Any) -> Any:
    add_flag_variables(dst["on"]["workflow_dispatch"]["inputs"])
    add_flag_variables(dst["on"]["workflow_call"]["inputs"])
    add_flag_env(dst["jobs"])

    return dst


def generate__notify(dst: Any) -> Any:
    add_flag_variables(dst["on"]["workflow_call"]["inputs"])
    add_flag_env(dst["jobs"])

    return dst


def generate__weekly(dst: Any) -> Any:
    cfg = config.get_config()

    all_jobs = []
    for name, weekly_cfg in cfg.weekly.items():
        for runner_nickname in weekly_cfg.runners:
            runner = runners.get_runner_by_nickname(runner_nickname)
            if runner.nickname == "unknown":
                raise ValueError(
                    f"Runner {runner_nickname} not found in bench_runner.toml"
                )
            job = {
                "uses": "./.github/workflows/_benchmark.yml",
                "needs": "determine_head",
                "with": {
                    "fork": "python",
                    "ref": "${{ needs.determine_head.outputs.commit }}",
                    "machine": runner.name,
                    "benchmarks": "all_and_excluded",
                    "pgo": True,
                    **flags.flags_to_gha_variables_yml(weekly_cfg.flags),
                },
                "secrets": "inherit",
            }
            job_name = f"weekly-{name}-{runner.nickname}"
            dst["jobs"][job_name] = job
            all_jobs.append(job_name)

    dst["jobs"]["generate"]["needs"].extend(all_jobs)

    return dst


def generate_generic(dst: Any) -> Any:
    return dst


GENERATORS = {
    "benchmark.src.yml": generate_benchmark,
    "_benchmark.src.yml": generate__benchmark,
    "_pystats.src.yml": generate__pystats,
    "_notify.src.yml": generate__notify,
    "_weekly.src.yml": generate__weekly,
}


def _main(check: bool) -> None:
    WORKFLOW_PATH.mkdir(parents=True, exist_ok=True)

    # Copy the bench_runner.toml file first, because it might be needed by the
    # other steps

    if not (ROOT_PATH / "bench_runner.toml").is_file():
        shutil.copyfile(
            TEMPLATE_PATH / "bench_runner.toml",
            ROOT_PATH / "bench_runner.toml",
        )

    for src_path in TEMPLATE_PATH.glob("*"):
        if not src_path.is_file():
            continue

        print(f"Processing {src_path}...")

        if src_path.name.endswith(".src.yml"):
            dst_path = WORKFLOW_PATH / (src_path.name[:-8] + ".yml")
            generator = GENERATORS.get(src_path.name, generate_generic)
            src = load_yaml(src_path)
            dst = generator(src)
            write_yaml(dst_path, dst, check)
        elif src_path.name.endswith(".src.py"):
            dst_path = ROOT_PATH / (src_path.name[:-7] + ".py")
            write_python(dst_path, src_path.read_text(), check)
        else:
            dst_path = ROOT_PATH / src_path.name
            if not dst_path.is_file():
                if check:
                    fail_check(dst_path)
                else:
                    shutil.copyfile(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(
        description="Install the Github Actions and other files",
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether any files need regeneration",
    )
    args = parser.parse_args()

    _main(args.check)


if __name__ == "__main__":
    main()
