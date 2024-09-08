"""
Utilities to generate markdown tables.
"""

from pathlib import Path
from typing import Iterable, Sequence, TextIO
from urllib.parse import quote


def output_table(
    fd: TextIO, head: Sequence[str], rows: Sequence[Sequence[str]]
) -> None:
    """
    Output a table in markdown format.
    """

    def output_row(row):
        fd.write(f'| {" | ".join(row)} |\n')

    output_row(head)
    output_row(col.endswith(":") and "---:" or "---" for col in head)
    for row in rows:
        output_row(row)


def replace_section(filename: Path, name: str, content: str) -> None:
    """
    Replace a table in a markdown file with the new content.

    The section is defined with the delimiters:

    ```
    <!-- START {name} -->
    ... content goes here ...
    <!-- END {name} -->
    ```
    """
    lines = iter(filename.read_text().splitlines())

    with filename.open("w") as fd:
        for line in lines:
            if line == f"<!-- START {name} -->":
                fd.write(line + "\n")
                fd.write(content)
                fd.write("\n")

                for line in lines:
                    if line == f"<!-- END {name} -->":
                        fd.write(line + "\n")
                        break
            else:
                fd.write(line + "\n")


def md_link(text: str, link: str, root: Path | None = None) -> str:
    """
    Formats a Markdown link. The link is resolved relative to the given root.
    """
    if root is not None:
        link = str(Path(link).resolve().relative_to(root.parent.resolve()))
    if not str(link).startswith("http"):
        link = "/".join(quote(x) for x in Path(link).parts)
    return f"[{text}]({link})"


def link_to_hash(hash: str, fork: str) -> str:
    """
    Create a markdown link to a specific hash of a specific fork on GitHub.
    """
    return md_link(
        hash,
        f"https://github.com/{fork}/cpython/commit/{hash}",
    )


def write_md_list(fd: TextIO, entries: Iterable[str]) -> None:
    """
    Writes a markdown list.
    """
    for val in entries:
        fd.write(f"- {val}\n")
    fd.write("\n")


def write_details(fd: TextIO, summary: str, lines: Iterable[str]) -> None:
    """
    Writes a <details> section.
    """
    fd.write("<details>\n")
    fd.write(f"<summary>{summary}</summary>\n\n")
    for line in lines:
        fd.write(line)
        fd.write("\n")
    fd.write("\n</details>\n\n")
