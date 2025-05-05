import argparse
import os


import rich_argparse


from bench_runner import config
from bench_runner import flags as mflags
from bench_runner import gh


def generate_dirname(
    date: str, version: str, cpython_hash: str, flags: list[str]
) -> str:
    if len(flags):
        flag_string = [",".join(sorted(flags))]
    else:
        flag_string = []
    return "-".join(
        ["bm", date[:10].replace("-", ""), version, cpython_hash[:7], *flag_string]
    )


def _main(
    fork: str, ref: str, head: str, date: str, version: str, flags: list[str]
) -> None:
    dirname = generate_dirname(date, version, head, flags)
    actor = os.environ.get("GITHUB_ACTOR", "UNKNOWN")
    github_repo = os.environ.get("GITHUB_REPOSITORY", "UNKNOWN")

    lines = ["🤖 This is the friendly benchmarking bot with some new results!", ""]
    line = f"@{actor}: [{fork}/{ref}]"
    skip_publish = config.get_config().publish_mirror.skip

    if skip_publish:
        line += f"(https://github.com/{github_repo}/tree/main/results/{dirname})"
    else:
        line += f"(https://github.com/{github_repo}-public/tree/main/results/{dirname})"
    print(f"::notice ::{line}")
    lines.append(line)

    gh.send_notification("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Send a notification that the workflow is complete.",
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
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
