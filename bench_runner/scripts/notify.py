import argparse
import os


from bench_runner import gh
from bench_runner import util


def generate_dirname(
    date: str, version: str, cpython_hash: str, tier2: bool, jit: bool, nogil: bool
) -> str:
    flags = util.get_flags(tier2, jit, nogil)
    return "-".join(
        ["bm", date[:10].replace("-", ""), version, cpython_hash[:7], *flags]
    )


def _main(
    fork: str,
    ref: str,
    head: str,
    date: str,
    version: str,
    tier2: bool,
    jit: bool,
    nogil: bool,
) -> None:
    dirname = generate_dirname(date, version, head, tier2, jit, nogil)
    actor = os.environ.get("GITHUB_ACTOR", "UNKNOWN")
    github_repo = os.environ.get("GITHUB_REPOSITORY", "UNKNOWN")

    lines = ["🤖 This is the friendly benchmarking bot with some new results!", ""]
    line = (
        f"@{actor}: "
        f"[{fork}/{ref}]"
        f"(https://github.com/{github_repo}-public/tree/main/results/{dirname})"
    )
    print(f"::notice ::{line}")
    lines.append(line)

    gh.send_notification("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        """
        Send a notification that the workflow is complete.
        """
    )
    parser.add_argument("--fork")
    parser.add_argument("--ref")
    parser.add_argument("--head")
    parser.add_argument("--date")
    parser.add_argument("--version")
    parser.add_argument("--tier2")
    parser.add_argument("--jit")
    parser.add_argument("--nogil")
    args = parser.parse_args()

    _main(
        args.fork,
        args.ref,
        args.head,
        args.date,
        args.version,
        args.tier2 != "false",
        args.jit != "false",
        args.nogil != "false",
    )


if __name__ == "__main__":
    main()
