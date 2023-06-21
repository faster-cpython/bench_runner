"""
Regenerates some Github Actions workflow files from templates.
"""

import argparse
import copy
import io
from pathlib import Path
import shutil
import sys
from typing import Any


from ruamel.yaml import YAML


from bench_runner import runners


ROOT_PATH = Path()
TEMPLATE_PATH = Path(__file__).parent.parent / "templates"
WORKFLOW_PATH = Path() / ".github" / "workflows"


def fail_check(dst: Path):
    print(f"{dst.relative_to(ROOT_PATH)} needs to be regenerated.")
    print("Run `python -m bench_runner.scripts.install` and commit the result.")
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


def generate__benchmark(input_path: Path, output_path: Path, check: bool) -> None:
    """
    Generates _benchmark.yml from _benchmark.src.yml.

    For each runner machine, inserts the platform-specific set of steps for
    that machine.

    Inserts the list of available machines to the drop-down presented to the
    user.
    """
    available_runners = [r for r in runners.get_runners() if r.available]
    runner_choices = [x.name for x in available_runners] + ["all"]

    src = load_yaml(input_path)
    dst = copy.deepcopy(src)

    dst["jobs"] = {}
    for runner in available_runners:
        runner_template = copy.deepcopy(src["jobs"][f"benchmark-{runner.os}"])
        runner_template["runs-on"].append(runner.name)
        runner_template[
            "if"
        ] = f"${{{{ (inputs.machine == '{runner.name}' || inputs.machine == 'all') }}}}"
        dst["jobs"][f"benchmark-{runner.name}"] = runner_template

    dst["on"]["workflow_dispatch"]["inputs"]["machine"]["options"] = runner_choices

    write_yaml(output_path, dst, check)


def generate_benchmark(input_path: Path, output_path: Path, check: bool) -> None:
    """
    Generates benchmark.yml from benchmark.src.yml.

    Inserts the list of available machines to the drop-down presented to the
    user.
    """
    available_runners = [r for r in runners.get_runners() if r.available]
    runner_choices = [x.name for x in available_runners] + ["all"]

    src = load_yaml(input_path)
    src["on"]["workflow_dispatch"]["inputs"]["machine"]["options"] = runner_choices
    write_yaml(output_path, src, check)


def generate_generic(input_path: Path, output_path: Path, check: bool) -> None:
    if check:
        input_content = input_path.read_bytes()
        output_content = output_path.read_bytes()
        if input_content != output_content:
            fail_check(output_path)
    else:
        shutil.copyfile(input_path, output_path)


GENERATORS = {
    "benchmark.src.yml": generate_benchmark,
    "_benchmark.src.yml": generate__benchmark,
}


def main(check: bool) -> None:
    WORKFLOW_PATH.mkdir(parents=True, exist_ok=True)

    for path in TEMPLATE_PATH.glob("*.src.yml"):
        generator = GENERATORS.get(path.name, generate_generic)
        generator(path, WORKFLOW_PATH / (path.name[:-8] + ".yml"), check)

    for path in TEMPLATE_PATH.glob("*"):
        if path.name.endswith(".src.yml"):
            continue

        if not (ROOT_PATH / path.name).is_file():
            if check:
                fail_check(ROOT_PATH / path.name)
            else:
                shutil.copyfile(path, ROOT_PATH / path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Install the Github Actions and other files")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether any files need regeneration",
    )
    args = parser.parse_args()

    main(args.check)
