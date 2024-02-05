import argparse
from pathlib import Path


from bench_runner import git
from bench_runner.result import has_result
from bench_runner import util


def _main(
    need_to_run: bool,
    machine: str,
    pystats: bool,
    tier2: bool,
    jit: bool,
    cpython: Path = Path("cpython"),
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
            flags = []
            if tier2:
                flags.extend(util.TIER2_FLAGS)
            if jit:
                flags.extend(util.JIT_FLAGS)

            need_to_run = (
                machine == "all"
                or has_result(
                    Path("results"),
                    merge_base,
                    machine,
                    pystats,
                    flags,
                    util.get_benchmark_hash(),
                )
                is None
            )

            print(f"ref={merge_base}")
            print(f"need_to_run={str(need_to_run).lower()}")


def main():
    parser = argparse.ArgumentParser(
        """
        Find the git merge-base in CPython main of a given commit, and also
        determine whether we already have data for that commit.
        """
    )
    parser.add_argument("need_to_run")
    parser.add_argument("machine")
    parser.add_argument("pystats")
    parser.add_argument("tier2")
    parser.add_argument("jit")
    args = parser.parse_args()

    _main(
        args.need_to_run != "false",
        args.machine,
        args.pystats != "false",
        args.tier2 != "false",
        args.jit != "false",
    )


if __name__ == "__main__":
    main()
