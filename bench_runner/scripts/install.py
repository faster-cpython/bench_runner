"""
Regenerates some Github Actions workflow files from templates.
"""

import argparse
import copy
import functools
import io
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any


from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString


from bench_runner import runners


ROOT_PATH = Path()
TEMPLATE_PATH = Path(__file__).parents[1] / "templates"
WORKFLOW_PATH = Path() / ".github" / "workflows"


@functools.cache
def get_requirements():
    with open("requirements.txt") as fd:
        return list(fd.readlines())[0].strip()


def fail_check(dst: Path):
    print(f"{dst.relative_to(ROOT_PATH)} needs to be regenerated.", file=sys.stderr)
    print(
        "Run `python -m bench_runner install` and commit the result.",
        file=sys.stderr,
    )
    sys.exit(1)


def write_yaml(dst: Path, contents: Any, check: bool):
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

    if check:
        if not dst.is_file():
            fail_check(dst)

        with open(dst) as fd:
            orig_contents = fd.read()
        fd = io.StringIO()
        do_write(contents, fd)
        new_contents = fd.getvalue()
        if orig_contents != new_contents:
            fail_check(dst)
    else:
        with open(dst, "w") as fd:
            do_write(contents, fd)


def load_yaml(src: Path) -> Any:
    """
    Load YAML from `src`.
    """
    with open(src) as fd:
        yaml = YAML()
        return yaml.load(fd)


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
        runner_template["if"] = (
            f"${{{{ (inputs.machine == '{runner.name}' || inputs.machine == 'all') }}}}"
        )
        dst["jobs"][f"benchmark-{runner.name}"] = runner_template

    dst["on"]["workflow_dispatch"]["inputs"]["machine"]["options"] = runner_choices

    return dst


def generate_benchmark(dst: Any) -> Any:
    """
    Generates benchmark.yml from benchmark.src.yml.

    Inserts the list of available machines to the drop-down presented to the
    user.
    """
    available_runners = [r for r in runners.get_runners() if r.available]
    runner_choices = [*[x.name for x in available_runners], "all"]

    dst["on"]["workflow_dispatch"]["inputs"]["machine"]["options"] = runner_choices

    dst["jobs"]["determine_base"]["steps"][2][
        "run"
    ] = f"pip install {shlex.quote(get_requirements())}"

    return dst


def generate__notify(dst: Any) -> Any:
    dst["jobs"]["notify"]["steps"][2][
        "run"
    ] = f"pip install {shlex.quote(get_requirements())}"

    return dst


def generate_generic(dst: Any) -> Any:
    return dst


GENERATORS = {
    "benchmark.src.yml": generate_benchmark,
    "_benchmark.src.yml": generate__benchmark,
    "_notify.src.yml": generate__notify,
}


def _main(check: bool) -> None:
    WORKFLOW_PATH.mkdir(parents=True, exist_ok=True)

    env = load_yaml(TEMPLATE_PATH / "env.yml")

    for src_path in TEMPLATE_PATH.glob("*.src.yml"):
        dst_path = WORKFLOW_PATH / (src_path.name[:-8] + ".yml")
        generator = GENERATORS.get(src_path.name, generate_generic)
        src = load_yaml(src_path)
        dst = generator(src)
        dst = {"env": env, **dst}
        write_yaml(dst_path, dst, check)

    for path in TEMPLATE_PATH.glob("*"):
        if path.name.endswith(".src.yml") or path.name == "env.yml":
            continue

        if not (ROOT_PATH / path.name).is_file():
            if check:
                fail_check(ROOT_PATH / path.name)
            else:
                shutil.copyfile(path, ROOT_PATH / path.name)


def main():
    parser = argparse.ArgumentParser("Install the Github Actions and other files")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether any files need regeneration",
    )
    args = parser.parse_args()

    _main(args.check)


if __name__ == "__main__":
    main()
