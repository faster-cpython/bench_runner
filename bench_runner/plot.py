from __future__ import annotations


import argparse
from collections import defaultdict
import datetime
import functools
import json
from pathlib import Path
import re
import tempfile
from typing import Callable, Iterable, Sequence


from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import rich_argparse
from scour import scour


matplotlib.use("agg")


from . import config as mconfig
from . import result
from . import runners as mrunners
from .util import PathLike


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


def savefig(output_filename: PathLike, **kwargs):
    class Options:
        quiet = True
        remove_descriptive_elements = True
        strip_comments = True
        indent_type = "none"
        strip_ids = True
        shorten_ids = True
        digits = 3

    output_filename = Path(output_filename)

    plt.savefig(output_filename, **kwargs)
    plt.close("all")

    if output_filename.suffix == ".svg":
        with tempfile.NamedTemporaryFile(
            dir=output_filename.parent, delete=False
        ) as tmp:
            with open(output_filename) as fd:
                scour.start(Options(), fd, tmp)
            output_filename.unlink()
            Path(tmp.name).rename(output_filename)


@functools.cache
def get_longitudinal_plot_config():
    cfg = mconfig.get_bench_runner_config()
    if "plot" in cfg:
        raise ValueError(
            "It looks like you have an old plot config in bench_runner.toml. "
            "See 'new plot config' in CHANGELOG.md for migration info."
        )

    plots = cfg.get("longitudinal_plot", {})
    subplots = plots.get("subplots", [])

    for subplot in subplots:
        assert "base" in subplot
        assert "version" in subplot
        if "flags" not in subplot:
            subplot["flags"] = []

    return plots


@functools.cache
def get_flag_effect_plot_config():
    cfg = mconfig.get_bench_runner_config()

    plots = cfg.get("flag_effect_plot", {})
    subplots = plots.get("subplots", [])

    for subplot in subplots:
        assert "name" in subplot
        assert "head_flags" in subplot
        subplot["head_flags"] = sorted(set(subplot["head_flags"]))
        if "base_flags" not in subplot:
            subplot["base_flags"] = []
        else:
            subplot["base_flags"] = sorted(set(subplot["base_flags"]))
        if "runner_map" not in subplot:
            subplot["runner_map"] = {}

    return plots


def plot_diff_pair(ax, data):
    if not len(data):
        return []

    all_data = []
    violins = []
    colors = []

    for i, (name, values, _mean) in enumerate(data):
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
    output_filename: PathLike,
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

    savefig(output_filename)


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


def longitudinal_plot(
    results: Iterable[result.Result],
    output_filename: PathLike,
    getter: Callable[
        [result.BenchmarkComparison], float | None
    ] = lambda r: r.geometric_mean_float,
    differences: tuple[str, str] = ("slower", "faster"),
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

    output_filename = Path(output_filename)
    all_cfg = get_longitudinal_plot_config().get("subplots", [])
    if len(all_cfg) == 0:
        print("No longitudinal plot config found. Skipping.")
        return

    data_cache = output_filename.with_suffix(".json")
    if data_cache.is_file():
        with data_cache.open() as fd:
            data = json.load(fd)
    else:
        data = {}

    axs: Sequence[matplotlib.Axes]  # pyright: ignore

    fig, _axs = plt.subplots(
        len(all_cfg),
        1,
        figsize=(10, 5 * len(all_cfg)),
        layout="constrained",
    )  # type: ignore
    if len(all_cfg) == 1:
        axs = [_axs]
    else:
        axs = _axs

    results = [r for r in results if r.fork == "python"]
    runners = mrunners.get_runners()

    for cfg, ax in zip(all_cfg, axs):
        version = [int(x) for x in cfg["version"].split(".")]
        assert len(version) == 2, "Version config should only be major.minor"
        ver_results = [
            r for r in results if list(r.parsed_version.release[0:2]) == version
        ]

        subtitle = f"Python {cfg['version']}.x vs. {cfg['base']}"
        if len(cfg["flags"]):
            subtitle += f" ({','.join(cfg['flags'])})"
        ax.set_title(subtitle)

        first_runner = True

        for runner in runners:
            runner_results = [
                r
                for r in ver_results
                if r.nickname == runner.nickname and set(r.flags) == set(cfg["flags"])
            ]

            for r in results:
                if (
                    r.nickname == runner.nickname
                    and r.version == cfg["base"]
                    and r.flags == []
                ):
                    ref = r
                    break
            else:
                continue

            runner_results.sort(
                key=lambda x: datetime.datetime.fromisoformat(x.commit_datetime)
            )
            dates = [
                datetime.datetime.fromisoformat(x.commit_datetime)
                for x in runner_results
            ]
            changes = [
                get_comparison_value(ref, r, cfg["base"]) for r in runner_results
            ]

            if any(x is not None for x in changes):
                ax.plot(
                    dates,
                    changes,
                    color=runner.plot.color,
                    linestyle=runner.plot.style,
                    marker=runner.plot.marker,
                    markersize=5,
                    label=runner.plot.name,
                    alpha=0.9,
                )

            if first_runner:
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
                first_runner = False

        annotate_y_axis(ax, differences)

    fig.suptitle(title)

    savefig(output_filename, dpi=150)

    with data_cache.open("w") as fd:
        json.dump(data, fd, indent=2)


def _standardize_xlims(axs: Sequence[matplotlib.Axes]) -> None:  # pyright: ignore
    if not len(axs):
        return

    minx = None
    maxx = None
    for ax in axs:
        if not ax.has_data():
            continue
        data = ax.get_lines()[0].get_data()[0]
        if len(data) > 1:
            xlim = ax.get_xlim()
            if minx is None or maxx is None:
                minx, maxx = xlim
            else:
                minx = min(minx, xlim[0])
                maxx = max(maxx, xlim[1])

    for ax in axs:
        if ax.has_data():
            ax.set_xlim((minx, maxx))


def flag_effect_plot(
    results: Iterable[result.Result],
    output_filename: PathLike,
    getter: Callable[
        [result.BenchmarkComparison], float | None
    ] = lambda r: r.geometric_mean_float,
    differences: tuple[str, str] = ("slower", "faster"),
    title="Performance improvement by configuration",
):
    output_filename = Path(output_filename)

    # We don't need to track the performance of the Tier 2 configuration

    subplots = get_flag_effect_plot_config().get("subplots", [])
    if len(subplots) == 0:
        print("No flag effect plot config found. Skipping.")
        return

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
        with data_cache.open() as fd:
            data = json.load(fd)
    else:
        data = {}

    axs: Sequence[matplotlib.Axes]  # pyright: ignore

    fig, _axs = plt.subplots(
        len(subplots), 1, figsize=(10, 5 * len(subplots)), layout="constrained"
    )  # type: ignore
    if len(subplots) == 1:
        axs = [_axs]
    else:
        axs = _axs

    results = [r for r in results if r.fork == "python"]

    commits: dict[str, dict[tuple[str, ...], dict[str, result.Result]]] = {}
    for r in results:
        commits.setdefault(r.nickname, {}).setdefault(tuple(r.flags), {})[
            r.cpython_hash
        ] = r

    runner_groups = defaultdict(list)
    runners = mrunners.get_runners()
    for runner in runners:
        runner_groups[runner.hostname].append(runner)

    for subplot, ax in zip(subplots, axs):
        ax.set_title(f"Effect of {subplot['name']}")

        for runner in runners:
            head_results = commits.get(runner.nickname, {}).get(
                tuple(subplot["head_flags"]), {}
            )
            base_results = commits.get(
                subplot["runner_map"].get(runner.nickname, runner.nickname), {}
            ).get(tuple(subplot["base_flags"]), {})

            line = []
            for cpython_hash, r in head_results.items():
                if cpython_hash in base_results:
                    line.append(
                        (
                            r.commit_datetime,
                            get_comparison_value(base_results[cpython_hash], r),
                        )
                    )
            line.sort(key=lambda x: datetime.datetime.fromisoformat(x[0]))

            dates = [datetime.datetime.fromisoformat(x[0]) for x in line]
            changes = [x[1] for x in line]

            if any(x is not None for x in changes):
                ax.plot(
                    dates,
                    changes,
                    color=runner.plot.color,
                    linestyle=runner.plot.style,
                    marker=runner.plot.marker,
                    markersize=5,
                    label=runner.plot.name,
                    alpha=0.9,
                )

        annotate_y_axis(ax, differences)

    fig.suptitle(title)

    _standardize_xlims(axs)

    savefig(output_filename, dpi=150)

    with data_cache.open("w") as fd:
        json.dump(data, fd, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Compare two benchmark .json files",
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument("ref", help="The reference .json file")
    parser.add_argument("head", help="The head .json file")
    parser.add_argument("output", help="Output filename")
    args = parser.parse_args()

    ref = result.Result.from_filename(Path(args.ref))
    head = result.Result.from_filename(Path(args.head))
    compare = result.BenchmarkComparison(ref, head, "base")

    compare.write_timing_plot(Path(args.output))
