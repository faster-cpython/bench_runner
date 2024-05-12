from __future__ import annotations


import argparse
import datetime
from operator import attrgetter
from pathlib import Path
import re
from typing import Callable, Iterable


from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import ujson


matplotlib.use("agg")


from . import result


INTERPRETER_HEAVY = {
    "chaos",
    "coroutines",
    "deepcopy",
    "deltablue",
    "generators",
    "go",
    "hexiom",
    "logging",
    "nbody",
    "pickle_pure_python",
    "pprint",
    "raytrace",
    "richards",
    "richards_super",
    "sqlglot_parse",
    "tomli_loads",
    "unpack_sequence",
    "unpickle_pure_python",
}


def plot_diff_pair(ax, data):
    if not len(data):
        return []

    all_data = []
    violins = []
    colors = []

    for i, (name, values, mean) in enumerate(data):
        if values is not None:
            idx = np.round(np.linspace(0, len(values) - 1, 100)).astype(int)
            violins.append(values[idx])
            all_data.extend(values)
            if name in INTERPRETER_HEAVY:
                colors.append("red")
            else:
                colors.append("C0")
        else:
            violins.append([1.0])
            all_data.extend([1.0])
            colors.append("C0")
            ax.text(1.01, i + 1, "insignificant")

    violins.append(all_data)

    violin = ax.violinplot(
        violins,
        vert=False,
        showmeans=True,
        showmedians=False,
        widths=1.0,
        quantiles=[[0.1, 0.9]] * len(violins),
    )

    violin["cquantiles"].set_linestyle(":")
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)

    for i, values in enumerate(violins):
        if not np.all(values == [1.0]):
            mean = np.mean(values)
            ax.text(mean, i + 1.3, f"{mean:.04f}", size=8)

    return all_data


def formatter(val, pos):
    return f"{val:.02f}×"


def plot_diff(
    combined_data: result.CombinedData,
    output_filename: Path,
    title: str,
    differences: tuple[str, str],
) -> None:
    _, axs = plt.subplots(
        figsize=(8, 2 + len(combined_data) * 0.3), layout="constrained"
    )
    plt.axvline(1.0)
    plot_diff_pair(axs, combined_data)
    names = [x[0] for x in combined_data]
    names.append("ALL")
    axs.set_yticks(np.arange(len(names)) + 1, names)
    axs.set_ylim(0, len(names) + 1)
    axs.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    axs.xaxis.set_major_formatter(formatter)
    xlim = axs.get_xlim()
    if xlim[0] > 0.75 and xlim[1] < 1.25:
        axs.set_xlim(0.75, 1.25)
    axs.annotate(
        f"{differences[1]} ⟶",
        xy=(1.0, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(10, 0),
        textcoords="offset pixels",
    )
    axs.annotate(
        f"⟵ {differences[0]}",
        xy=(1.0, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-10, 0),
        textcoords="offset pixels",
        horizontalalignment="right",
    )
    axs.grid()
    axs.set_title(title)

    plt.savefig(output_filename)
    plt.close()


def get_micro_version(version: str) -> str:
    micro = version.split(".")[-1].replace("+", "")
    if match := re.match(r"[0-9]+([a-z]+.+)", micro):
        micro = match.groups()[0]
    return micro


def annotate_y_axis(ax, differences: tuple[str, str]):
    ax.yaxis.set_major_formatter(formatter)
    ax.grid()
    ylim = ax.get_ylim()
    if ylim[0] > 0.9 and ylim[1] < 1.3:
        ax.set_ylim([0.9, 1.3])
    ax.legend(loc="upper left")
    ax.annotate(
        f"{differences[1]} ⟶",
        xy=(0.0, 1.0),
        xycoords=("axes fraction", "data"),
        xytext=(10, 10),
        textcoords="offset pixels",
        rotation=90,
        clip_on=True,
    )
    ax.annotate(
        f"⟵ {differences[0]}",
        xy=(0.0, 1.0),
        xycoords=("axes fraction", "data"),
        xytext=(10, -10),
        textcoords="offset pixels",
        rotation=90,
        clip_on=True,
        verticalalignment="top",
    )


tier2_date = datetime.datetime.fromisoformat("2023-11-11T00:00:00Z")
jit_date = datetime.datetime.fromisoformat("2024-01-31T00:00:00Z")


def filter_by_config(results: Iterable[result.Result]) -> list[result.Result]:
    correct_results = []
    for r in results:
        dt = datetime.datetime.fromisoformat(r.commit_datetime)
        if dt < tier2_date:
            if not r.is_tier2 and not r.is_jit:
                correct_results.append(r)
        elif dt < jit_date:
            if r.is_tier2 and not r.is_jit:
                correct_results.append(r)
        else:
            if r.is_jit:
                correct_results.append(r)
    return correct_results


def add_axvline(ax, dt: datetime.datetime, name: str):
    ax.axvline(dt)
    ax.annotate(
        name,
        xy=(dt, 0.9),
        xycoords=("data", "axes fraction"),
        xytext=(10, 0),
        textcoords="offset pixels",
        rotation=90,
    )


# TODO: Make this configurable
def longitudinal_plot(
    results: Iterable[result.Result],
    output_filename: Path,
    bases=["3.10.4", "3.11.0", "3.12.0", "3.12.0"],
    runners=[
        "linux",
        "pythonperf2",
        "arminc",
        "darwin",
        "pythonperf1",
        "pythonperf1_win32",
    ],
    names=["linux", "linux2", "linux-aarch64", "macos", "win64", "win32"],
    colors=["C0", "C0", "C4", "C2", "C3", "C3"],
    styles=["-", ":", "--", "-", "-", ":"],
    versions=[(3, 11), (3, 12), (3, 13), (3, 13)],
    getter: Callable[
        [result.BenchmarkComparison], float | None
    ] = lambda r: r.hpt_percentile_float(99),
    differences: tuple[str, str] = ("slower", "faster"),
    markers=["s", "s", "s", "^", ".", "."],
    title="Performance improvement by configuration",
):
    def get_comparison_value(ref, r, base):
        key = ",".join((str(ref.filename)[8:], str(r.filename)[8:], base))
        if key in data:
            return data[key]
        else:
            value = getter(result.BenchmarkComparison(ref, r, base))
            data[key] = value
            return value

    data_cache = output_filename.with_suffix(".json")
    if data_cache.is_file():
        with open(data_cache) as fd:
            data = ujson.load(fd)
    else:
        data = {}

    fig, axs = plt.subplots(
        len(versions), 1, figsize=(10, 5 * len(versions)), layout="constrained"
    )

    results = [r for r in results if r.fork == "python"]

    for i, (version, base, ax) in enumerate(zip(versions, bases, axs)):
        version_str = ".".join(str(x) for x in version)
        ver_results = [r for r in results if r.parsed_version.release[0:2] == version]

        subtitle = f"Python {version_str}.x vs. {base}"
        if i == 3:
            subtitle += " (with JIT)"
        ax.set_title(subtitle)

        for runner_i, (runner, name, color, style, marker) in enumerate(
            zip(runners, names, colors, styles, markers)
        ):
            runner_results = [r for r in ver_results if r.nickname == runner]

            # For 3.13, only use Tier 2 results after 2023-11-11 and JIT results
            # after 2024-01-30
            if i == 2:
                runner_results = [
                    r for r in runner_results if r.flags == []
                ]
            elif i == 3:
                runner_results = filter_by_config(runner_results)

            for r in results:
                if r.nickname == runner and r.version == base:
                    ref = r
                    break
            else:
                continue

            runner_results.sort(key=attrgetter("commit_datetime"))
            dates = [
                datetime.datetime.fromisoformat(x.commit_datetime)
                for x in runner_results
            ]
            changes = [get_comparison_value(ref, r, base) for r in runner_results]

            if any(x is not None for x in changes):
                ax.plot(
                    dates,
                    changes,
                    color=color,
                    linestyle=style,
                    marker=marker,
                    markersize=5,
                    label=name,
                    alpha=0.9,
                )

            if runner_i > 0:
                continue

            annotations = set()
            for r, date, change in zip(runner_results, dates, changes):
                micro = get_micro_version(r.version)
                if micro not in annotations and not r.version.endswith("+"):
                    annotations.add(micro)
                    text = ax.annotate(
                        micro,
                        xy=(date, change),
                        xycoords="data",
                        xytext=(-3, 15),
                        textcoords="offset points",
                        rotation=90,
                        arrowprops=dict(arrowstyle="-", connectionstyle="arc"),
                    )
                    text.set_color("#888")
                    text.set_size(8)
                    text.arrow_patch.set_color("#888")

        annotate_y_axis(ax, differences)

        # Add a line for when Tier 2 and JIT were turned on
        if i == 3:
            add_axvline(ax, tier2_date, "TIER 2")
            add_axvline(ax, jit_date, "JIT")

    fig.suptitle(title)

    plt.savefig(output_filename, dpi=150)
    plt.close()

    with open(data_cache, "w") as fd:
        ujson.dump(data, fd, indent=2)


def flag_effect_plot(
    results: Iterable[result.Result],
    output_filename: Path,
    flags=["NOGIL", "JIT", "PYTHON_UOPS"],
    configs=["Free threading", "JIT compiler", "Tier 2 interpreter"],
    runners=[
        "linux",
        "pythonperf2",
        "arminc",
        "darwin",
        "pythonperf1",
        "pythonperf1_win32",
    ],
    names=["linux", "linux2", "linux-aarch64", "macos", "win64", "win32"],
    colors=["C0", "C0", "C4", "C2", "C3", "C3"],
    styles=["-", ":", "--", "-", "-", ":"],
    getter: Callable[
        [result.BenchmarkComparison], float | None
    ] = lambda r: r.hpt_percentile_float(99),
    differences: tuple[str, str] = ("slower", "faster"),
    markers=["s", "s", "s", "^", ".", "."],
    title="Performance improvement by configuration",
):
    def get_comparison_value(ref, r):
        key = ",".join((str(ref.filename)[8:], str(r.filename)[8:]))
        if key in data:
            return data[key]
        else:
            value = getter(result.BenchmarkComparison(ref, r, "default"))
            data[key] = value
            return value

    data_cache = output_filename.with_suffix(".json")
    if data_cache.is_file():
        with open(data_cache) as fd:
            data = ujson.load(fd)
    else:
        data = {}

    fig, axs = plt.subplots(
        len(flags), 1, figsize=(10, 5 * len(flags)), layout="constrained"
    )

    results = [r for r in results if r.fork == "python"]

    commits: dict[str, dict[str, dict[str, result.Result]]] = {}
    for r in results:
        for flag in flags:
            if flag in r.flags:
                break
        else:
            flag = ""
        commits.setdefault(r.nickname, {}).setdefault(flag, {})[r.cpython_hash] = r

    for config, flag, ax in zip(configs, flags, axs):
        ax.set_title(f"Effect of {config} vs. Tier 1 (same commit)")

        for runner, name, color, style, marker in zip(
            runners, names, colors, styles, markers
        ):
            runner_results = commits.get(runner, {})
            base_results = runner_results.get("", {})

            line = []
            for cpython_hash, r in runner_results.get(flag, {}).items():
                if cpython_hash in base_results:
                    line.append(
                        (
                            r.commit_datetime,
                            get_comparison_value(base_results[cpython_hash], r),
                        )
                    )
            line.sort()

            dates = [datetime.datetime.fromisoformat(x[0]) for x in line]
            changes = [x[1] for x in line]

            if any(x is not None for x in changes):
                ax.plot(
                    dates,
                    changes,
                    color=color,
                    linestyle=style,
                    marker=marker,
                    markersize=5,
                    label=name,
                    alpha=0.9,
                )

        annotate_y_axis(ax, differences)

    fig.suptitle(title)

    minx = min(ax.get_xlim()[0] for ax in axs[1:])
    maxx = max(ax.get_xlim()[1] for ax in axs[1:])
    for ax in axs:
        ax.set_xlim((minx, maxx))

    plt.savefig(output_filename, dpi=150)
    plt.close()

    with open(data_cache, "w") as fd:
        ujson.dump(data, fd, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compare two benchmark .json files")
    parser.add_argument("ref", help="The reference .json file")
    parser.add_argument("head", help="The head .json file")
    parser.add_argument("output", help="Output filename")
    args = parser.parse_args()

    ref = result.Result.from_filename(Path(args.ref))
    head = result.Result.from_filename(Path(args.head))
    compare = result.BenchmarkComparison(ref, head, "base")

    compare.write_timing_plot(Path(args.output))
