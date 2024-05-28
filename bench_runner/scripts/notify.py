import argparse
import os


from bench_runner import flags as mflags
from bench_runner import gh


def generate_dirname(
    date: str, version: str, cpython_hash: str, flags: list[str]
) -> str:
    return "-".join(
        ["bm", date[:10].replace("-", ""), version, cpython_hash[:7], *flags]
    )


def _main(
    fork: str, ref: str, head: str, date: str, version: str, flags: list[str]
) -> None:
    dirname = generate_dirname(date, version, head, flags)
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
    parser.add_argument("--flags")
    args = parser.parse_args()

    _main(
        args.fork,
        args.ref,
        args.head,
        args.date,
        args.version,
        mflags.parse_flags(args.flags),
    )


if __name__ == "__main__":
    main()
