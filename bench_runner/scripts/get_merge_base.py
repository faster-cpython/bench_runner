from pathlib import Path
import sys


from bench_runner import git
from bench_runner.result import has_result


def main(
    need_to_run: bool, machine: str, pystats: bool, cpython: Path = Path("cpython")
) -> None:
    if not need_to_run:
        print("ref=xxxxxxx")
        print("need_to_run=false")
    else:
        merge_base = git.get_git_merge_base(cpython)

        if merge_base is None:
            print("ref=xxxxxxx")
            print("need_to_run=false")
        else:
            need_to_run = (
                machine == "all"
                or has_result(Path("results"), merge_base, machine, pystats) is None
            )

            print(f"ref={merge_base}")
            print(f"need_to_run={str(need_to_run).lower()}")


if __name__ == "__main__":
    need_to_run = sys.argv[-3] != "false"
    pystats = sys.argv[-1] != "false"

    main(need_to_run, sys.argv[-2], pystats)
