# Utilities to manage a results file
from __future__ import annotations


from collections import defaultdict
import functools
import io
import json
from operator import itemgetter
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import unquote


import numpy as np
from packaging import version
import pyperf


from . import bases as mbases
from . import config
from . import flags as mflags
from . import git
from . import hpt
from . import plot
from . import runners
from . import util
from .util import PathLike


CombinedData = list[tuple[str, np.ndarray | None, float]]


@functools.lru_cache(maxsize=100)
def _load_contents(filename: Path) -> dict[str, Any]:
    with filename.open("rb") as fd:
        return json.load(fd)


def clear_contents_cache() -> None:
    _load_contents.cache_clear()


def _clean(string: str) -> str:
    """
    Clean an arbitrary string to be suitable for a filename.

    It can't contain dashes, since dashes are used as a delimiter.
    """
    return string.replace("-", "_")


def _clean_for_url(string: str) -> str:
    """
    Clean an arbitrary string to be suitable for a filename, being careful to
    create something that can be re-used as a URL.

    It can't contain dashes, since dashes are used as a delimiter.
    """
    return string.replace("-", "%2d")


def _get_platform_value(python: PathLike, item: str) -> str:
    """
    Get a value from the platform module of the given Python interpreter.
    """
    output = subprocess.check_output(
        [python, "-c", f"import platform; print(platform.{item}())"], encoding="utf-8"
    )
    return output.strip().lower()


def _get_architecture(python: PathLike) -> str:
    machine = _get_platform_value(python, "machine")
    bits = eval(_get_platform_value(python, "architecture"))[0]
    if bits == "32bit":
        return {"x86_64": "x86", "amd64": "x86", "arm64": "arm32"}.get(machine, machine)
    return machine


class Comparison:
    def __init__(
        self, ref: "Result", head: "Result", base: str, force_valid: bool = False
    ):
        self.ref = ref
        self.head = head
        self.base = base
        self.valid_comparison = force_valid or not (
            self.ref == self.head
            or (
                self.ref.cpython_hash == self.head.cpython_hash
                and self.ref.flags == self.head.flags
            )
        )

    def copy(self):
        return type(self)(self.ref, self.head, self.base)

    @property
    def base_filename(self) -> Path | None:
        if not self.valid_comparison:
            return None

        return self.head.filename.parent / (
            f"{self.head.filename.stem}-vs-{self.base}.txt"
        )

    def get_files(self) -> Iterable[tuple[Callable, str, str]]:
        return []


class BenchmarkComparison(Comparison):
    def get_files(self) -> Iterable[tuple[Callable, str, str]]:
        if self.base_filename is None:
            return
        yield (self.write_table, ".md", "table")
        yield (self.write_timing_plot, ".svg", "time plot")
        if not self.head.is_windows() and self.base == "base":
            yield (self.write_memory_plot, "-mem.svg", "memory plot")

    @functools.cached_property
    def _contents(self) -> str | None:
        if self.base_filename is None:
            return None

        if self.base_filename.with_suffix(".md").is_file():
            with self.base_filename.with_suffix(".md").open(encoding="utf-8") as fd:
                return fd.read()
        else:
            return self._generate_contents()

    @property
    def _contents_lines(self) -> list[str]:
        if self._contents is None:
            return []
        else:
            return self._contents.splitlines()

    def _generate_contents(self) -> str:
        fd = io.StringIO()
        fd.write(
            subprocess.check_output(
                [
                    "pyperf",
                    "compare_to",
                    "-G",
                    "--table",
                    "--table-format",
                    "md",
                    self.ref.filename,
                    self.head.filename,
                ],
                encoding="utf-8",
            )
        )
        fd.write("\n")
        fd.write(
            "- Geometric mean (including insignificant results): "
            f"{self._calculate_geometric_mean()}"
        )
        fd.write("\n\n")
        fd.write(hpt.make_report(self.ref.filename, self.head.filename))
        fd.write("\n")
        fd.write("# Memory\n")
        fd.write(f"- memory change: {self._calculate_memory_change()}")
        return fd.getvalue()

    def write_table(self, filename: PathLike) -> str | None:
        filename = Path(filename)

        entries = [
            ("fork", unquote(self.head.fork)),
            ("ref", self.head.ref),
            ("machine", f"{self.head.system}-{self.head.machine}"),
            ("commit hash", self.head.cpython_hash),
            ("commit date", self.head.commit_date),
            ("overall geometric mean", self.geometric_mean),
            ("HPT reliability", self.hpt_reliability),
            ("HPT 99th percentile", self.hpt_percentile(99)),
        ]

        if self.memory_change is not None:
            entries.append(("Memory change", self.memory_change))

        contents = self._contents
        assert contents is not None

        with filename.open("w") as fd:
            fd.write(f"# Results vs. {self.base}\n\n")
            for key, val in entries:
                fd.write(f"- {key}: {val}\n")
            fd.write("\n")
            fd.write(contents)

    def _get_combined_data(
        self, ref_data: dict[str, np.ndarray], head_data: dict[str, np.ndarray]
    ) -> CombinedData:
        def remove_outliers(values, m=2):
            return values[
                abs(values - np.mean(values)) < np.multiply(m, np.std(values))
            ]

        def calculate_diffs(ref_values, head_values) -> tuple[np.ndarray | None, float]:
            if len(ref_values) > 3 and len(head_values) > 3:
                sig, t_score = pyperf._utils.is_significant(ref_values, head_values)
                if not sig:
                    return None, 0.0
                else:
                    ref_values = remove_outliers(ref_values)
                    head_values = remove_outliers(head_values)
            values = np.outer(ref_values, 1.0 / head_values).flatten()
            values.sort()
            return values, float(values.mean())

        cfg = config.get_config()
        excluded = cfg.benchmarks.excluded_benchmarks
        combined_data = []
        for name, ref in ref_data.items():
            if len(ref) != 0 and name in head_data and name not in excluded:
                head = head_data[name]
                if len(ref) == len(head):
                    combined_data.append((name, *calculate_diffs(ref, head)))
        combined_data.sort(key=itemgetter(2))
        return combined_data

    def get_timing_diff(self) -> CombinedData:
        ref_data = self.ref.get_timing_data()
        head_data = self.head.get_timing_data()
        return self._get_combined_data(ref_data, head_data)

    def write_timing_plot(self, filename: PathLike) -> None:
        plot.plot_diff(
            self.get_timing_diff(),
            filename,
            (
                "Timings of "
                f"{unquote(self.head.fork)}-{self.head.ref}-"
                f"{self.head.cpython_hash}"
                f" vs. {self.ref.version}"
            ),
            ("slower", "faster"),
        )

    def get_memory_diff(self) -> CombinedData:
        ref_data = self.ref.get_memory_data()
        head_data = self.head.get_memory_data()
        # Explicitly reversed so higher is bigger
        return self._get_combined_data(head_data, ref_data)

    def write_memory_plot(self, filename: PathLike) -> None:
        plot.plot_diff(
            self.get_memory_diff(),
            filename,
            (
                "Memory usage of "
                f"{unquote(self.head.fork)}-{self.head.ref}-"
                f"{self.head.cpython_hash}"
                f" vs. {self.ref.version}"
            ),
            ("less", "more"),
        )

    @functools.cached_property
    def geometric_mean_float(self) -> float | None:
        if not self.valid_comparison:
            return None

        data = self.get_timing_diff()

        product = np.prod(np.array([x[2] for x in data if x[1] is not None]))
        return float(product ** (1.0 / len(data)))

    def _calculate_geometric_mean(self) -> str:
        gm = self.geometric_mean_float
        if gm is None or gm == 1.0:
            return "not sig"
        elif gm > 1.0:
            return f"{gm:.03f}x faster"
        else:
            return f"{1.0+(1.0-gm):.03f}x slower"

    @property
    def geometric_mean(self) -> str:
        if not self.valid_comparison:
            return ""

        lines = self._contents_lines

        for line in lines[::-1]:
            if "Geometric mean (including insignificant results)" in line:
                geometric_mean = line.split(":", maxsplit=1)[-1].strip()
                break
        else:
            geometric_mean = "not sig"

        return geometric_mean

    def _calculate_memory_change(self):
        # Windows doesn't have memory data
        if self.head.is_windows():
            return "unknown"

        combined_data = self.get_memory_diff()
        points = np.array([x[2] for x in combined_data if x[1] is not None])
        if len(points) == 0:
            return "unknown"
        else:
            change = np.mean(points)
            return f"{change:.02f}x"

    @functools.cached_property
    def memory_change(self) -> str | None:
        if not self.valid_comparison:
            return ""

        for line in self._contents_lines:
            if line.startswith("- memory change:"):
                return line[line.find(":") + 1 :].strip()

        return None

    @property
    def memory_change_float(self) -> float | None:
        memory_change = self.memory_change
        if memory_change in (None, "unknown"):
            return None
        else:
            return float(memory_change.strip().strip("x"))

    @property
    def memory_summary(self) -> str:
        memory_change = self.memory_change or "unknown"
        return f"Memory usage: {memory_change}"

    @functools.cached_property
    def hpt_reliability(self) -> str | None:
        if not self.valid_comparison:
            return ""

        lines = self._contents_lines

        for line in lines:
            m = re.match(r"- Reliability score: (\S+)", line)
            if m is not None:
                return m.group(1)

        return None

    def hpt_percentile(self, percentile: int) -> str | None:
        if not self.valid_comparison:
            return ""

        lines = self._contents_lines

        for line in lines:
            m = re.match(r"- ([0-9]+)% likely to have a (\S+) of (\S+)", line)
            if m is not None:
                if int(m.group(1)) == percentile:
                    if m.group(2) == "slowdown":
                        suffix = "slower"
                    else:
                        suffix = "faster"
                    return f"{m.group(3)} {suffix}"

        return None

    def hpt_percentile_float(self, percentile: int) -> float | None:
        result = self.hpt_percentile(percentile)
        if result is not None:
            num = float(result.split()[0][:-1])
            if result.endswith("slower"):
                return 1.0 - (num - 1.0)
            else:
                return num
        else:
            return None

    @property
    def summary(self) -> str:
        if not self.valid_comparison:
            return ""

        result = self.geometric_mean
        result = result.replace("faster", "↑")
        result = result.replace("slower", "↓")

        return result

    @property
    def long_summary(self) -> str:
        if not self.valid_comparison:
            return ""

        result = f"Geometric mean: {self.geometric_mean}"
        reliability = self.hpt_reliability
        if reliability is not None:
            subresult = f"HPT: reliability of {reliability}"
            percentile = self.hpt_percentile(99)
            if percentile is not None:
                subresult += f", {percentile} at 99th %ile"
            result += f" ({subresult})"

        return result


class PystatsComparison(Comparison):
    def get_files(self) -> Iterable[tuple[Callable, str, str]]:
        if self.base_filename is None:
            return
        yield (self.write_pystats_diff, ".md", "pystats diff")

    def write_pystats_diff(self, filename: PathLike) -> None:
        filename = Path(filename)

        try:
            contents = subprocess.check_output(
                [
                    sys.executable,
                    Path("cpython") / "Tools" / "scripts" / "summarize_stats.py",
                    self.ref.filename,
                    self.head.filename,
                ],
                encoding="utf-8",
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            return None
        filename.write_text(contents)


def comparison_factory(ref: "Result", head: "Result", base: str) -> Comparison:
    if head.result_info[0] == "raw results":
        return BenchmarkComparison(ref, head, base)
    elif head.result_info[0] == "pystats raw":
        return PystatsComparison(ref, head, base)

    raise ValueError(f"Can not compare result of type {head.result_info[0]}")


class Result:
    """
    Stores information about a single result file.
    """

    def __init__(
        self,
        nickname: str,
        machine: str,
        fork: str,
        ref: str,
        version: str,
        cpython_hash: str,
        extra: Iterable[str] | None = None,
        suffix: str = ".json",
        commit_datetime: str | None = None,
        flags: Iterable[str] | None = None,
    ):
        self.nickname = nickname
        self.machine = machine
        self.fork = fork
        self.ref = ref
        self.version = version
        self.cpython_hash = cpython_hash
        self.extra = extra or []
        self.suffix = suffix
        self.flags = sorted(set(flags or []))
        self._commit_datetime = commit_datetime
        self._filename = None
        self.bases = {}

    @classmethod
    def from_filename(cls, filename: PathLike) -> "Result":
        filename = Path(filename)
        (
            name,
            _,
            nickname,
            machine,
            fork,
            ref,
            version,
            cpython_hash,
            *extra,
        ) = filename.stem.split("-")
        assert name == "bm"
        (name, _, _, _, *flags) = filename.parent.name.split("-")
        assert name == "bm"
        assert len(flags) <= 1
        if len(flags) == 1:
            flags = flags[0].split(",")
        obj = cls(
            nickname=nickname,
            machine=machine,
            fork=fork,
            ref=ref,
            version=version,
            cpython_hash=cpython_hash,
            extra=extra,
            suffix=filename.suffix,
            flags=flags,
        )
        obj._filename = filename
        return obj

    @classmethod
    def from_arbitrary_filename(cls, filename: PathLike) -> "Result":
        filename = Path(filename)
        content = json.loads(filename.read_text())
        obj = cls(
            nickname="unknown",
            machine="unknown",
            fork="unknown",
            ref=filename.stem,
            version="unknown",
            cpython_hash=content.get("metadata", {}).get("commit_id", "unknown"),
            extra=[],
            suffix=filename.suffix,
            flags=[],
        )
        obj._filename = filename
        return obj

    @classmethod
    def from_scratch(
        cls,
        python: PathLike,
        fork: str,
        ref: str,
        extra: Iterable[str] | None = None,
        flags: Iterable[str] | None = None,
    ) -> "Result":
        result = cls(
            _clean(runners.get_nickname_for_hostname()),
            _clean(_get_architecture(python)),
            _clean_for_url(fork),
            _clean(ref[:20]),
            _clean(_get_platform_value(python, "python_version")),
            git.get_git_hash(Path("cpython"))[:7],
            extra or [],
            ".json",
            commit_datetime=git.get_git_commit_date(Path("cpython")),
            flags=flags or [],
        )
        return result

    @property
    def filename(self) -> Path:
        if self._filename is None:
            date = self.commit_date.replace("-", "")
            if self.extra:
                extra = ["-".join(self.extra)]
            else:
                extra = []
            if self.flags:
                flags = [",".join(self.flags)]
            else:
                flags = []
            self._filename = (
                Path("results")
                / "-".join(["bm", date, self.version, self.cpython_hash, *flags])
                / (
                    "-".join(
                        [
                            "bm",
                            date,
                            self.nickname,
                            self.machine,
                            self.fork,
                            self.ref,
                            self.version,
                            self.cpython_hash,
                            *extra,
                        ]
                    )
                    + self.suffix
                )
            )
        return self._filename

    @functools.cached_property
    def result_info(self) -> tuple[str | None, str | None, str | None]:
        match (self.extra, self.suffix):
            case ([], ".json"):
                return ("raw results", None, None)
            case (["pystats", "vs", base], ".md"):
                return ("pystats diff", base, None)
            case (["pystats", *benchmark], ".md"):
                if benchmark:
                    benchmark = benchmark[0]
                else:
                    benchmark = None
                return ("pystats table", None, benchmark)
            case (["pystats", *benchmark], ".json"):
                if benchmark:
                    benchmark = benchmark[0]
                else:
                    benchmark = None
                return ("pystats raw", None, benchmark)
            case (["vs", base], ".md"):
                return ("table", base, None)
            case (["vs", base], ".svg"):
                return ("time plot", base, None)
            case (["vs", base, "mem"], ".svg"):
                return ("memory plot", base, None)
        raise ValueError(
            f"Unknown result type (extra={self.extra} suffix={self.suffix})"
        )

    @property
    def contents(self) -> dict[str, Any]:
        return _load_contents(self.filename)

    @functools.cached_property
    def metadata(self) -> dict[str, Any]:
        return self.contents.get("metadata", {})

    @functools.cached_property
    def commit_datetime(self) -> str:
        if self._commit_datetime is not None:
            return self._commit_datetime
        return self.metadata.get("commit_date", "<unknown>")

    @property
    def commit_date(self) -> str:
        return self.commit_datetime[:10]

    @functools.cached_property
    def run_datetime(self) -> str:
        return self.contents["benchmarks"][0]["runs"][0]["metadata"]["date"]

    @property
    def run_date(self) -> str:
        return self.run_datetime[:10]

    @property
    def commit_merge_base(self) -> str | None:
        return self.metadata.get("commit_merge_base", None)

    @property
    def benchmark_hash(self) -> str | None:
        return self.metadata.get("benchmark_hash", None)

    @property
    def hostname(self) -> str:
        return self.metadata.get("hostname", "unknown host")

    @property
    def system(self) -> str:
        return runners.get_runner_by_nickname(self.nickname).os

    def is_windows(self) -> bool:
        if self.nickname != "unknown":
            return self.system == "windows"
        else:
            return (
                self.metadata.get("platform", "unknown").lower().startswith("windows")
            )

    @property
    def runner(self) -> str:
        return f"{self.system} {self.machine} ({self.nickname})"

    @property
    def cpu_model_name(self) -> str:
        return self.metadata.get("cpu_model_name", "missing")

    @property
    def platform(self) -> str:
        return self.metadata.get("platform", "missing")

    @property
    def github_action_url(self) -> str | None:
        return self.metadata.get("github_action_url", None)

    @property
    def hash_and_flags(self) -> str:
        # A representation for the user that combines the commit hash and other flags
        return " ".join(
            [self.cpython_hash, *(f"({x})" for x in mflags.flags_to_human(self.flags))]
        )

    @functools.cached_property
    def benchmark_names(self) -> set[str]:
        names = set()
        for benchmark in self.contents["benchmarks"]:
            if "metadata" in benchmark:
                names.add(benchmark["metadata"]["name"])
            else:
                names.add(self.contents["metadata"]["name"])
        return names

    @functools.cached_property
    def parsed_version(self):
        from packaging import version as pkg_version

        return pkg_version.parse(self.version.replace("+", "0"))

    def get_timing_data(self) -> dict[str, np.ndarray]:
        cfg = config.get_config()
        data = {}
        excluded = cfg.benchmarks.excluded_benchmarks

        for benchmark in self.contents["benchmarks"]:
            name = benchmark.get("metadata", self.contents["metadata"])["name"]
            if name not in excluded:
                row = []
                for run in benchmark["runs"]:
                    row.extend(run.get("values", []))
                data[name] = np.array(row, dtype=np.float64)

        return data

    def get_memory_data(self) -> dict[str, np.ndarray]:
        cfg = config.get_config()
        data = {}
        excluded = cfg.benchmarks.excluded_benchmarks

        # On MacOS, there was a bug in pyperf where the `mem_max_rss` value was
        # erroneously multiplied by 1024.  (BSD defines maxrss in bytes, Linux
        # in kilobytes).

        needs_correction = self.system == "darwin" and version.parse(
            self.contents["metadata"]["perf_version"]
        ) < version.parse("2.6.3")

        def memory_value(metadata):
            if mem := metadata.get("command_max_rss"):
                return mem
            elif mem := metadata.get("mem_max_rss"):
                if needs_correction:
                    mem /= 1024
                return mem
            return None

        for benchmark in self.contents["benchmarks"]:
            metadata = benchmark.get("metadata", self.contents["metadata"])
            name = metadata["name"]
            if name not in excluded:
                if mem := memory_value(metadata):
                    data[name] = np.array([mem], dtype=np.float64)
                else:
                    row = []
                    for run in benchmark["runs"]:
                        metadata = run.get("metadata", {})
                        if mem := memory_value(metadata):
                            row.append(mem)
                    data[name] = np.array(row, dtype=np.float64)

        return data


def has_result(
    results_dir: PathLike,
    commit_hash: str,
    machine: str,
    pystats: bool,
    flags: Sequence[str],
    benchmark_hash: str,
    progress: bool = True,
) -> Result | None:
    if machine in ("__really_all", "all"):
        nickname = None
    else:
        _, _, nickname = machine.split("-")

    results = load_all_results([], results_dir, False, progress=progress)

    if pystats:
        for result in results:
            if (
                commit_hash.startswith(result.cpython_hash)
                and result.result_info[0] == "pystats raw"
                and result.flags == flags
                and result.benchmark_hash == benchmark_hash
            ):
                return result
    else:
        for result in results:
            if (
                commit_hash.startswith(result.cpython_hash)
                and (nickname is None or result.nickname == nickname)
                and result.flags == flags
                and result.benchmark_hash == benchmark_hash
            ):
                return result

    return None


def match_to_bases(
    results: Iterable[Result], bases: Sequence[str] | None, progress: bool = True
):
    def find_match(result, candidates, base, func):
        # Try for an exact match (same benchmark_hash) first,
        # then fall back to less exact.
        for result_set in [
            candidates.get(result.benchmark_hash, []),
            *(v for k, v in candidates.items() if k != result.benchmark_hash),
        ]:
            for ref in result_set:
                if ref != result and func(ref):
                    result.bases[base] = comparison_factory(ref, result, base)
                    return True
        return False

    if bases is None:
        bases = []

    if progress:
        track = util.track  # type: ignore
    else:

        def track(it, *_args, **_kwargs):
            return it

    groups = defaultdict(lambda: defaultdict(list))
    for result in track(results, "Loading results"):
        if result.fork == "python":
            groups[(result.nickname, tuple(result.extra))][
                result.benchmark_hash
            ].append(result)

    for result in track(results, "Matching results to bases"):
        candidates = groups[(result.nickname, tuple(result.extra))]

        if (
            result.version not in bases
            and result.parsed_version.release[0:2]
            < mbases.get_minimum_version_for_all_comparisons()
        ):
            continue

        if bases is not None:
            for base in bases:
                find_match(
                    result,
                    candidates,
                    base,
                    lambda ref: ref.version == base and ref.flags == [],
                )

        merge_base = result.commit_merge_base
        found_base = False
        if merge_base is not None:
            _merge_base: str = merge_base
            found_base = find_match(
                result,
                candidates,
                "base",
                lambda ref: (
                    _merge_base.startswith(ref.cpython_hash)
                    and ref.flags == result.flags
                ),
            )

            for flag in config.get_config().bases.compare_to_default:
                if result.flags == [flag]:
                    found_default_base = find_match(
                        result,
                        candidates,
                        "default_base_vs_" + flag,
                        lambda ref: (
                            _merge_base.startswith(ref.cpython_hash) and ref.flags == []
                        ),
                    )
                    found_base = found_base or found_default_base

        if not found_base and result.fork == "python" and result.flags != []:
            # Compare builds with flags with builds with no flags
            find_match(
                result,
                candidates,
                "base",
                lambda ref: (
                    ref.cpython_hash == result.cpython_hash and ref.flags == []
                ),
            )


def load_all_results(
    bases: Sequence[str] | None,
    results_dir: PathLike,
    sorted: bool = True,
    match: bool = True,
    progress: bool = True,
) -> list[Result]:
    results = []

    for entry in Path(results_dir).glob("**/*.json"):
        result = Result.from_filename(entry)
        if result.result_info[0] not in ["raw results", "pystats raw"]:
            continue
        results.append(result)
    if len(results) == 0:
        raise ValueError("Didn't find any results.  That seems fishy.")

    if match:
        match_to_bases(results, bases, progress=progress)

    if sorted:
        results.sort(
            key=lambda x: (
                x.parsed_version,
                x.commit_datetime,
                tuple(x.flags),
                x.filename,  # Just to produce a stable ordering
            ),
            reverse=True,
        )

    return results
