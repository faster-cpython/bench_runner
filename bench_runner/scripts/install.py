"""
Regenerates some Github Actions workflow files from templates.
"""

import argparse
import copy
import functools
import importlib.metadata
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
WORKFLOW_PATH = ROOT_PATH / ".github" / "workflows"
PAGES_PATH = ROOT_PATH / "pages"


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


def write_with_check(
    dst: PathLike, contents: Any, check: bool, writer: Callable[[Any, TextIO], None]
) -> None:
    """
    Write `contents` to `dst` using the writer function `writer`.

    If `check` is True, raise SystemExit if the file would change. This is used
    in CI to confirm that the file was regenerated after changes to the source
    file.
    """
    dst = Path(dst)

    if check:
        if not dst.is_file():
            fail_check(dst)

        orig_contents = dst.read_text()
        fd = io.StringIO()
        writer(contents, fd)
        new_contents = fd.getvalue()
        if orig_contents != new_contents:
            fail_check(dst)
    else:
        with dst.open("w") as fd:
            writer(contents, fd)


def write_yaml(dst: PathLike, contents: Any, check: bool) -> None:
    """
    Write `contents` to `dst` as YAML.

    If `check` is True, raise SystemExit if the file would change. This is used
    in CI to confirm that the file was regenerated after changes to the source
    file.
    """

    def do_write(contents, fd):
        fd.write("# Generated file: !!! DO NOT EDIT !!!\n")
        fd.write("---\n")
        yaml = YAML()
        yaml.dump(contents, fd)

    return write_with_check(dst, contents, check, do_write)


def load_yaml(src: PathLike) -> Any:
    """
    Load YAML from `src`.
    """
    with Path(src).open() as fd:
        yaml = YAML()
        return yaml.load(fd)


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
    available_runners = [r for r in runners.get_runners() if r.available]
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


def get_skip_publish_mirror() -> bool:
    return config.get_bench_runner_config().get("publish_mirror", {}).get("skip", False)


def get_webui_enabled() -> bool:
    return config.get_bench_runner_config().get("webui", {}).get("enabled", False)


def generate_benchmark(dst: Any) -> Any:
    """
    Generates benchmark.yml from benchmark.src.yml.

    Inserts the list of available machines to the drop-down presented to the
    user.
    """
    available_runners = [r for r in runners.get_runners() if r.available]
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

    if get_skip_publish_mirror():
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


def generate_static(dst: Any) -> Any:
    if get_webui_enabled():
        return dst
    return None


def generate_generic(dst: Any) -> Any:
    return dst


GENERATORS = {
    "benchmark.src.yml": generate_benchmark,
    "_benchmark.src.yml": generate__benchmark,
    "_pystats.src.yml": generate__pystats,
    "_notify.src.yml": generate__notify,
}


def write_pages(check: bool) -> None:
    if not get_webui_enabled():
        return

    def get_writer(suffix: str) -> Callable[[Any, TextIO], None]:
        comment = {".py": ("#", ""), ".js": ("//", ""), ".html": ("<!--", "-->")}[
            suffix
        ]

        def do_write(contents, fd):
            for key, value in vars.items():
                contents = contents.replace("{{" + key + "}}", value)

            fd.write(
                f"{comment[0]} Generated file: !!! DO NOT EDIT !!!{comment[1]}\n\n"
            )
            fd.write(contents)

        return do_write

    webui_config = config.get_bench_runner_config().get("webui", {})

    vars = {
        "bench_runner_version": importlib.metadata.version("bench_runner"),
        "public_org": webui_config.get("public_org", ""),
        "public_repo": webui_config.get("public_repo", ""),
    }

    PAGES_PATH.mkdir(exist_ok=True)

    for path in (TEMPLATE_PATH / "pages").glob("*"):
        dst_path = PAGES_PATH / path.name
        if path.suffix == ".png":
            shutil.copyfile(path, dst_path)
        else:
            write_with_check(dst_path, path.read_text(), check, get_writer(path.suffix))


def _main(check: bool) -> None:
    WORKFLOW_PATH.mkdir(parents=True, exist_ok=True)

    env = load_yaml(TEMPLATE_PATH / "env.yml")

    for path in TEMPLATE_PATH.glob("*"):
        if path.is_dir() or path.name.endswith(".src.yml") or path.name == "env.yml":
            continue

        if not (ROOT_PATH / path.name).is_file():
            if check:
                fail_check(ROOT_PATH / path.name)
            else:
                shutil.copyfile(path, ROOT_PATH / path.name)

    for src_path in TEMPLATE_PATH.glob("*.src.yml"):
        dst_path = WORKFLOW_PATH / (src_path.name[:-8] + ".yml")
        generator = GENERATORS.get(src_path.name, generate_generic)
        src = load_yaml(src_path)
        dst = generator(src)
        dst = {"env": env, **dst}
        write_yaml(dst_path, dst, check)

    write_pages(check)


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
