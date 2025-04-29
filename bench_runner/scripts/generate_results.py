from __future__ import annotations


import argparse
from collections import defaultdict
import datetime
import io
from pathlib import Path
import sys
from typing import Iterable, TextIO, Sequence
from urllib.parse import unquote


import rich
import rich_argparse


from bench_runner.bases import get_bases
from bench_runner import flags as mflags
from bench_runner import plot
from bench_runner.result import (
    load_all_results,
    Result,
)
from bench_runner import table
from bench_runner import util
from bench_runner.util import PathLike


def _tuple_to_nested_dicts(entries: Iterable[tuple], d: dict | None = None) -> dict:
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


def save_generated_results(results: Iterable[Result], force: bool = False) -> None:
    """
    Write out the comparison tables and plots for every result.

    By default, files are only written out if they don't already exist. To force
    regeneration, pass ``force=True``.
    """
    work = []
    for result in results:
        for compare in result.bases.values():
            if compare.valid_comparison:
                for func, suffix, _ in compare.get_files():
                    filename = util.apply_suffix(compare.base_filename, suffix)
                    if filename.exists() and force:
                        filename.unlink()

                    if not filename.exists():
                        work.append((func, filename))

    assert len(list(set(x[1] for x in work))) == len(work)

    rich.print(f"Generating {len(work)} derived results")

    for func, filename in util.track(
        work,
        "Generating results",
    ):
        func(filename)


def output_results_index(
    fd: TextIO, bases: Iterable[str], results: Iterable[Result], filename: PathLike
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
                entry = [compare.summary, "<br>"]
                for _, suffix, file_type in compare.get_files():
                    entry.append(
                        table.md_link(
                            util.TYPE_TO_ICON[file_type],
                            str(util.apply_suffix(compare.base_filename, suffix)),
                            filename,
                        )
                    )
                versus.append("".join(entry))
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
        try:
            idx = order.index(val.split()[0])
        except ValueError:
            idx = -1
        return idx, val

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
    results: Iterable[Result], n_recent: int = 3, days: int = 3
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


def get_most_recent_pystats(
    results: Iterable[Result],
) -> Iterable[tuple[list[str], Result]]:
    for flags in ([], ["PYTHON_UOPS"], ["JIT"]):
        candidate_pystats = [
            result
            for result in results
            if result.result_info == ("pystats raw", None, None)
            and result.fork == "python"
            and result.flags == flags
        ]
        if len(candidate_pystats):
            yield (
                flags,
                sorted(
                    candidate_pystats,
                    key=lambda x: (x.parsed_version, x.commit_datetime),
                    reverse=True,
                )[0],
            )


def generate_index(
    filename: PathLike,
    bases: Iterable[str],
    all_results: Iterable[Result],
    benchmarking_results: Iterable[Result],
    summarize: bool = False,
) -> None:
    """
    Generate the tables, by each platform.
    """
    content = io.StringIO()

    for flags, result in get_most_recent_pystats(all_results):
        link = table.md_link(
            f"Most recent {','.join(flags)} pystats on main ({result.cpython_hash})",
            str(util.apply_suffix(result.filename, ".md")),
            filename,
        )
        content.write(f"- {link}\n")
    content.write("\n")

    for runner, results in results_by_runner(benchmarking_results):
        content.write(f"## {runner}\n")
        if summarize:
            results = summarize_results(results)
        output_results_index(content, bases, results, filename)
        content.write("\n")
    table.replace_section(filename, "table", content.getvalue())


def generate_indices(
    bases: Iterable[str],
    all_results: Iterable[Result],
    benchmarking_results: Iterable[Result],
    repo_dir: PathLike,
) -> None:
    """
    Generate both indices:

    - The summary one in `./README.md`
    - The full one in `./RESULTS.md`

    (For the ideas repo, the second file is at `results/README.md`).
    """
    repo_dir = Path(repo_dir)
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
    results: Iterable[Result],
) -> list[tuple[Path, str | None, str | None, str]]:
    entries = []
    dirpaths: set[Path] = set()
    refs = defaultdict(set)
    for result in results:
        dirpath = result.filename.parent
        dirpaths.add(dirpath)
        refs[dirpath].add(result.ref)
        entries.append(
            (dirpath, None, None, f"fork: {unquote(result.fork)}/{unquote(result.ref)}")
        )
        entries.append((dirpath, None, None, f"version: {result.version}"))
        config = ",".join(mflags.flags_to_human(result.flags))
        entries.append((dirpath, None, None, f"config: {config}"))
        link = table.link_to_hash(result.cpython_hash, result.fork)
        entries.append((dirpath, None, None, f"commit hash: {link}"))
        entries.append((dirpath, None, None, f"commit date: {result.commit_datetime}"))
        if result.commit_merge_base is not None:
            link = table.link_to_hash(result.commit_merge_base, "python")
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
            type, base, benchmark = result.result_info
            if type is not None and benchmark is None:
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


def generate_directory_indices(results: Iterable[Result]) -> None:
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

    for dirpath, dirresults in util.track(structure.items(), "Generating indices"):
        with (dirpath / "README.md").open("w") as fd:
            fd.write("# Results\n\n")
            table.write_md_list(fd, dirresults[None][None])
            for runner in sort_runner_names(dirresults.keys()):
                if runner is None:
                    continue
                data = dirresults[runner]
                fd.write(f"## {runner}\n\n")
                table.write_md_list(fd, data.get(None, ""))
                for base, subdata in data.items():
                    if base is None:
                        continue
                    fd.write(f"### vs. {base}\n\n")
                    table.write_md_list(fd, subdata)


def filter_broken_memory_results(results):
    return [r for r in results if r.nickname != "darwin"]


def _main(repo_dir: PathLike, force: bool = False, bases: Sequence[str] | None = None):
    repo_dir = Path(repo_dir)
    results_dir = repo_dir / "results"
    if bases is None:
        bases = get_bases()
    if len(bases) == 0:
        raise ValueError("Must have at least one base specified")
    rich.print(f"Comparing to bases: {','.join(bases)}")
    results = load_all_results(bases, results_dir)
    rich.print(f"Found {len(results)} results")
    save_generated_results(results, force=force)
    benchmarking_results = [r for r in results if r.result_info[0] == "raw results"]
    generate_indices(bases, results, benchmarking_results, repo_dir)
    generate_directory_indices(benchmarking_results)

    memory_benchmarking_results = filter_broken_memory_results(benchmarking_results)

    for plot_func, args, kwargs in util.track(
        [
            (
                plot.longitudinal_plot,
                (benchmarking_results, repo_dir / "longitudinal.svg"),
                {},
            ),
            (
                plot.flag_effect_plot,
                (benchmarking_results, repo_dir / "configs.svg"),
                {},
            ),
            (
                plot.longitudinal_plot,
                (memory_benchmarking_results, repo_dir / "memory_long.svg"),
                dict(
                    getter=lambda r: r.memory_change_float,
                    differences=("less", "more"),
                    title="Memory usage change by major version",
                ),
            ),
            (
                plot.flag_effect_plot,
                (memory_benchmarking_results, repo_dir / "memory_configs.svg"),
                dict(
                    getter=lambda r: r.memory_change_float,
                    differences=("less", "more"),
                    title="Memory usage change by configuration",
                ),
            ),
            (
                plot.benchmark_longitudinal_plot,
                (benchmarking_results, repo_dir / "benchmarks.svg"),
                {},
            ),
        ],
        "Generating plots",
    ):
        plot_func(*args, **kwargs)  # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description="""
        Generate index tables and comparison plots for all of the results.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
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
        rich.print(f"[red]{args.repo_dir} is not a directory.[/red]")
        sys.exit(1)

    _main(args.repo_dir, force=args.force)


if __name__ == "__main__":
    main()
