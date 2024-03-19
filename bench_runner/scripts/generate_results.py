from __future__ import annotations


import argparse
from collections import defaultdict
import datetime
import io
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Iterable, Optional, TextIO
from urllib.parse import unquote


from bench_runner.bases import get_bases
from bench_runner import gh
from bench_runner import plot
from bench_runner.result import (
    load_all_results,
    Comparison,
    Result,
)
from bench_runner import table
from bench_runner import util


def _tuple_to_nested_dicts(entries: list[tuple], d: Optional[dict] = None) -> dict:
    def recurse(entry: tuple, d: dict):
        if len(entry) == 2:
            d.setdefault(entry[0], [])
            if entry[1] not in d[entry[0]]:
                d[entry[0]].append(entry[1])
        else:
            recurse(entry[1:], d.setdefault(entry[0], {}))

    assert len(set(len(x) for x in entries)) == 1

    if d is None:
        d = {}

    for entry in entries:
        recurse(entry, d)
    return d


class ResultWriter:
    @staticmethod
    def need_to_generate(compare: Comparison) -> bool:
        return False

    @staticmethod
    def generate(filename: Path, compare) -> None:
        pass


class MarkdownResultWriter(ResultWriter):
    @staticmethod
    def need_to_generate(compare: Comparison) -> bool:
        return compare.contents is not None

    @staticmethod
    def generate(filename: Path, compare) -> None:
        if filename.exists():
            filename.unlink()
            compare = compare.copy()

        entries = [
            ("fork", unquote(compare.head.fork)),
            ("ref", compare.head.ref),
            ("machine", f"{compare.head.system}-{compare.head.machine}"),
            ("commit hash", compare.head.cpython_hash),
            ("commit date", compare.head.commit_date),
            ("overall geometric mean", compare.geometric_mean),
            ("HPT reliability", compare.hpt_reliability),
            ("HPT 99th percentile", compare.hpt_percentile(99)),
        ]

        if compare.memory_change is not None:
            entries.append(("Memory change", compare.memory_change))

        with open(filename, "w") as fd:
            fd.write(f"# Results vs. {compare.base}\n\n")
            for key, val in entries:
                fd.write(f"- {key}: {val}\n")
            fd.write("\n")
            fd.write(compare.contents or "")


class TimePlotResultWriter(ResultWriter):
    @staticmethod
    def need_to_generate(compare: Comparison) -> bool:
        return True

    @staticmethod
    def generate(filename: Path, compare) -> None:
        plot.plot_diff(
            compare.get_timing_diff(),
            filename,
            (
                "Timings of "
                f"{unquote(compare.head.fork)}-{compare.head.ref}-"
                f"{compare.head.cpython_hash}"
                f" vs. {compare.ref.version}"
            ),
            ("slower", "faster"),
        )


class MemoryPlotResultWriter(ResultWriter):
    @staticmethod
    def need_to_generate(compare: Comparison) -> bool:
        return compare.head.system != "windows" and compare.base == "base"

    @staticmethod
    def generate(filename: Path, compare) -> None:
        plot.plot_diff(
            compare.get_memory_diff(),
            filename,
            (
                "Memory usage of "
                f"{unquote(compare.head.fork)}-{compare.head.ref}-"
                f"{compare.head.cpython_hash}"
                f" vs. {compare.ref.version}"
            ),
            ("less", "more"),
        )


class PystatsDiffResultWriter(ResultWriter):
    @staticmethod
    def need_to_generate(compare: Comparison) -> bool:
        return compare.contents is not None

    @staticmethod
    def generate(filename: Path, compare) -> None:
        if filename.exists():
            filename.unlink()
            compare = compare.copy()

        with open(filename, "w") as fd:
            fd.write(compare.contents or "")


RESULT_TYPES = {
    "raw results": {
        ".md": MarkdownResultWriter,
        ".png": TimePlotResultWriter,
        "-mem.png": MemoryPlotResultWriter,
    },
    "pystats raw": {".md": PystatsDiffResultWriter},
    None: {},
}


def _worker(args) -> None:
    writer, output_filename, compare = args
    writer.generate(output_filename, compare)


def save_generated_results(results: Iterable[Result], force: bool = False) -> None:
    """
    Write out the comparison tables and plots for every result.

    By default, files are only written out if they don't already exist. To force
    regeneration, pass ``force=True``.
    """
    work = []
    directories_affected = set()
    people_affected = defaultdict(set)
    for result in results:
        for compare in result.bases.values():
            if compare.filename is not None:
                for suffix, writer in RESULT_TYPES[result.result_info[0]].items():
                    filename = compare.filename.parent / (
                        compare.filename.stem + suffix
                    )
                    if writer.need_to_generate(compare) and (
                        not filename.exists() or force
                    ):
                        work.append((writer, filename, compare))
                        directories_affected.add(filename.parent)
                        actor = compare.head.metadata.get("github_actor")
                        if actor is not None:
                            people_affected[actor].add(filename.parent)

    with multiprocessing.Pool() as pool:
        for i, _ in enumerate(pool.imap_unordered(_worker, work)):
            print(f"{i + 1:04d}/{len(work):04d}", end="\r")
        print()

    github_repo = os.environ.get("GITHUB_REPOSITORY", "UNKNOWN")
    for directory in directories_affected:
        print(
            "::notice ::New results at "
            f"https://github.com/{github_repo}-public/tree/main/{directory}"
        )

    if len(people_affected):
        send_notification(people_affected)


def send_notification(people_affected):
    github_repo = os.environ.get("GITHUB_REPOSITORY", "UNKNOWN")

    body = "🤖 This is the friendly benchmarking bot with some new results!\n\n"
    for actor, directories in people_affected.items():
        body += f"@{actor}: "
        for directory in directories:
            body += f"[New results](https://github.com/{github_repo}-public/tree/main/{directory}) "
        body += "\n"

    gh.send_notification(body)


def output_results_index(
    fd: TextIO, bases: list[str], results: Iterable[Result], filename: Path
):
    """
    Outputs a results index table.
    """
    bases = [*bases, "base"]

    head = ["date", "fork/ref", "hash/flags"] + [f"vs. {base}:" for base in bases]

    rows = []
    for result in results:
        versus = []
        for base in bases:
            if base in result.bases and result.bases[base].valid_comparison:
                compare = result.bases[base]
                entry = (
                    compare.summary
                    + "<br>"
                    + table.md_link(
                        util.TYPE_TO_ICON["table"],
                        compare.filename.with_suffix(".md"),
                        filename,
                    )
                    + table.md_link(
                        util.TYPE_TO_ICON["time plot"],
                        compare.filename.with_suffix(".png"),
                        filename,
                    )
                )
                if base == "base" and compare.memory_change not in (None, "unknown"):
                    entry += table.md_link(
                        util.TYPE_TO_ICON["memory plot"],
                        compare.filename.parent / (compare.filename.stem + "-mem.png"),
                        filename,
                    )
                versus.append(entry)
            else:
                versus.append("")

        rows.append(
            [
                table.md_link(
                    result.commit_date, str(result.filename.parent), filename
                ),
                f"{unquote(result.fork)}/{result.ref}",
                result.hash_and_flags,
                *versus,
            ]
        )
    table.output_table(fd, head, rows)


def sort_runner_names(runner_names: Iterable[str]) -> list[str]:
    # We want linux first, as the most meaningful/reliable one
    order = ["linux", "windows", "darwin"]

    def sorter(val):
        if val is None:
            return ()
        return order.index(val.split()[0]), val

    return sorted(runner_names, key=sorter)


def results_by_runner(
    results: Iterable[Result],
) -> Iterable[tuple[str, Iterable[Result]]]:
    """
    Separate results by the runner used.
    """
    by_runner = defaultdict(list)
    for result in results:
        if result.result_info[0] != "raw results":
            continue
        by_runner[result.runner].append(result)

    for runner_name in sort_runner_names(by_runner.keys()):
        yield (runner_name, by_runner[runner_name])


def summarize_results(
    results: Iterable[Result], bases: list[str], n_recent: int = 3, days: int = 3
) -> Iterable[Result]:
    """
    Create a shorter list of results which includes:

    - The 3 most recent
    - Any results in the last 3 days
    """
    results = list(results)
    new_results = []
    earliest = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    for i, result in enumerate(results):
        if i < n_recent or result.run_date >= earliest:
            new_results.append(result)
    return new_results


def get_most_recent_pystats(results: Iterable[Result]) -> Optional[Result]:
    candidate_pystats = [
        result
        for result in results
        if result.result_info[0] == "pystats raw" and result.fork == "python"
    ]
    if len(candidate_pystats):
        return sorted(
            candidate_pystats,
            key=lambda x: (x.commit_datetime, len(x.flags)),
            reverse=True,
        )[0]


def generate_index(
    filename: Path,
    bases: list[str],
    all_results: Iterable[Result],
    benchmarking_results: Iterable[Result],
    summarize: bool = False,
) -> None:
    """
    Generate the tables, by each platform.
    """
    content = io.StringIO()

    if (most_recent_pystats := get_most_recent_pystats(all_results)) is not None:
        link = table.md_link(
            f"Most recent pystats on main ({most_recent_pystats.cpython_hash})",
            str(most_recent_pystats.filename.with_suffix(".md")),
            filename,
        )
        content.write(f"{link}\n\n")

    for runner, results in results_by_runner(benchmarking_results):
        content.write(f"## {runner}\n")
        if summarize:
            results = summarize_results(results, bases)
        output_results_index(content, bases, results, filename)
        content.write("\n")
    table.replace_section(filename, "table", content.getvalue())


def generate_indices(
    bases: list[str],
    all_results: Iterable[Result],
    benchmarking_results: Iterable[Result],
    repo_dir: Path,
) -> None:
    """
    Generate both indices:

    - The summary one in `./README.md`
    - The full one in `./RESULTS.md`

    (For the ideas repo, the second file is at `results/README.md`).
    """
    generate_index(
        repo_dir / "README.md", bases, all_results, benchmarking_results, True
    )
    results_file = repo_dir / "RESULTS.md"
    if not results_file.is_file():
        results_file = repo_dir / "results" / "README.md"
    generate_index(results_file, bases, all_results, benchmarking_results, False)


def find_different_benchmarks(head: Result, ref: Result) -> tuple[list[str], list[str]]:
    head_benchmarks = head.benchmark_names
    base_benchmarks = ref.benchmark_names
    return (
        sorted(base_benchmarks - head_benchmarks),
        sorted(head_benchmarks - base_benchmarks),
    )


def get_directory_indices_entries(
    results: list[Result],
) -> list[tuple[Path, Optional[str], Optional[str], str]]:
    entries = []
    dirpaths: set[Path] = set()
    refs = defaultdict(set)
    for result in results:
        dirpath = result.filename.parent
        dirpaths.add(dirpath)
        refs[dirpath].add(result.ref)
        entries.append((dirpath, None, None, f"fork: {unquote(result.fork)}"))
        entries.append((dirpath, None, None, f"version: {result.version}"))
        entries.append((dirpath, None, None, f"tier 2: {result.is_tier2}"))
        entries.append((dirpath, None, None, f"jit: {result.is_jit}"))
        link = table.link_to_hash(result.cpython_hash, result.fork)
        entries.append((dirpath, None, None, f"commit hash: {link}"))
        entries.append((dirpath, None, None, f"commit date: {result.commit_datetime}"))
        if result.commit_merge_base is not None:
            link = table.link_to_hash(result.commit_merge_base, result.fork)
            entries.append((dirpath, None, None, f"commit merge base: {link}"))
        if result.github_action_url is not None:
            link = table.md_link("GitHub Action run", result.github_action_url)
            entries.append((dirpath, result.runner, None, link))

        entries.append(
            (dirpath, result.runner, None, f"cpu model: {result.cpu_model_name}")
        )
        entries.append((dirpath, result.runner, None, f"platform: {result.platform}"))

        if result.result_info[0] == "raw results":
            for base, compare in result.bases.items():
                entries.append((dirpath, result.runner, base, compare.long_summary))
                entries.append((dirpath, result.runner, base, compare.memory_summary))
                missing_benchmarks, new_benchmarks = find_different_benchmarks(
                    result, compare.ref
                )
                if len(missing_benchmarks):
                    prefix = base == "base" and "🔴 " or ""
                    entries.append(
                        (
                            dirpath,
                            result.runner,
                            base,
                            "missing benchmarks: "
                            f"{prefix}{', '.join(missing_benchmarks)}",
                        )
                    )
                if len(new_benchmarks):
                    entries.append(
                        (
                            dirpath,
                            result.runner,
                            base,
                            f"new benchmarks: {', '.join(new_benchmarks)}",
                        )
                    )

    for dirpath in dirpaths:
        entries.append(
            (dirpath, None, None, f"ref: {', '.join(sorted(refs[dirpath]))}")
        )

        for filename in sorted(list(dirpath.iterdir())):
            if filename.name == "README.md":
                continue
            result = Result.from_filename(filename)
            type, base = result.result_info
            if type is not None:
                entries.append(
                    (
                        dirpath,
                        result.runner,
                        base,
                        table.md_link(
                            util.TYPE_TO_ICON.get(type, "") + type, result.filename.name
                        ),
                    )
                )

    return entries


def generate_directory_indices(results: list[Result]) -> None:
    """
    Generate the indices that go in each results directory.
    """

    # The data is in a considerably different form than what we need to write
    # it out. Therefore, this first generates a list of tuples of the form:
    #    (dirpath, runner, base, entry)
    # then converts that to a nested dictionary and then writes it out to each
    # of the README.md files.

    entries = get_directory_indices_entries(results)
    structure = _tuple_to_nested_dicts(entries)

    for dirpath, dirresults in structure.items():
        util.status(".")
        with open(dirpath / "README.md", "w") as fd:
            fd.write("# Results\n\n")
            table.write_md_list(fd, dirresults[None][None])
            for runner in sort_runner_names(dirresults.keys()):
                if runner is None:
                    continue
                data = dirresults[runner]
                fd.write(f"## {runner}\n\n")
                table.write_md_list(fd, data[None])
                for base, subdata in data.items():
                    if base is None:
                        continue
                    fd.write(f"### vs. {base}\n\n")
                    table.write_md_list(fd, subdata)
    print()


def _main(repo_dir: Path, force: bool = False, bases: Optional[list[str]] = None):
    results_dir = repo_dir / "results"
    if bases is None:
        bases = get_bases()
    if len(bases) == 0:
        raise ValueError("Must have at least one base specified")
    print(f"Comparing to bases {bases}")
    results = load_all_results(bases, results_dir)
    print(f"Found {len(results)} results")
    print("Generating comparison results")
    save_generated_results(results, force=force)
    print("Generating indices")
    benchmarking_results = [r for r in results if r.result_info[0] == "raw results"]
    generate_indices(bases, results, benchmarking_results, repo_dir)
    generate_directory_indices(benchmarking_results)
    print("Generating longitudinal plot")
    plot.longitudinal_plot(benchmarking_results, repo_dir / "longitudinal.png")
    print("Generating configurations plot")
    plot.flag_effect_plot(benchmarking_results, repo_dir / "configs.png")
    print("Generating memory plots")
    plot.longitudinal_plot(
        benchmarking_results,
        repo_dir / "memory_long.png",
        getter=lambda r: r.memory_change_float,
        differences=("less", "more"),
        title="Memory usage change by major version",
    )
    plot.flag_effect_plot(
        benchmarking_results,
        repo_dir / "memory_configs.png",
        getter=lambda r: r.memory_change_float,
        differences=("less", "more"),
        title="Memory usage change by configuration",
    )


def main():
    parser = argparse.ArgumentParser(
        "Generate index tables and comparison plots for all of the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "repo_dir",
        nargs="?",
        type=Path,
        default=Path(),
        help="The location of the results repository",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the comparison files, even if they already exist.",
    )

    args = parser.parse_args()

    if not args.repo_dir.is_dir():
        print(f"{args.repo_dir} is not a directory.")
        sys.exit(1)

    _main(args.repo_dir, force=args.force)


if __name__ == "__main__":
    main()
