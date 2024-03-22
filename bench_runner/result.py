# Utilities to manage a results file
from __future__ import annotations


from collections import defaultdict
import functools
import io
from operator import itemgetter
from pathlib import Path
import re
import socket
import subprocess
import sys
from typing import Any, Callable, Iterable, Optional
from urllib.parse import unquote


import numpy as np
from packaging import version
import pyperf
import ujson


from . import git
from . import hpt
from . import plot
from . import runners


CombinedData = list[tuple[str, Optional[np.ndarray], float]]


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


def _get_platform_value(python: Path, item: str) -> str:
    """
    Get a value from the platform module of the given Python interpreter.
    """
    output = subprocess.check_output(
        [python, "-c", f"import platform; print(platform.{item}())"], encoding="utf-8"
    )
    return output.strip().lower()


def _get_architecture(python: Path) -> str:
    machine = _get_platform_value(python, "machine")
    bits = eval(_get_platform_value(python, "architecture"))[0]
    if bits == "32bit":
        return {"x86_64": "x86", "amd64": "x86", "arm64": "arm32"}.get(machine, machine)
    return machine


class Comparison:
    def __init__(self, ref: "Result", head: "Result", base: str):
        self.ref = ref
        self.head = head
        self.base = base
        self.valid_comparison = not (
            self.ref == self.head
            or (
                self.ref.cpython_hash == self.head.cpython_hash
                and self.ref.flags == self.head.flags
            )
        )

    def copy(self):
        return type(self)(self.ref, self.head, self.base)

    @property
    def base_filename(self) -> Optional[Path]:
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
        yield (self.write_timing_plot, ".png", "time plot")
        if self.head.system != "windows" and self.base == "base":
            yield (self.write_memory_plot, "-mem.png", "memory plot")

    @functools.cached_property
    def _contents(self) -> Optional[str]:
        if self.base_filename is None:
            return None

        if self.base_filename.with_suffix(".md").is_file():
            with open(
                self.base_filename.with_suffix(".md"), "r", encoding="utf-8"
            ) as fd:
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
        fd.write(hpt.make_report(self.ref.filename, self.head.filename))
        fd.write("\n")
        fd.write("# Memory\n")
        fd.write(f"- memory change: {self._calculate_memory_change()}")
        return fd.getvalue()

    def write_table(self, filename: Path) -> Optional[str]:
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

        with open(filename, "w") as fd:
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

        def calculate_diffs(
            ref_values, head_values
        ) -> tuple[Optional[np.ndarray], float]:
            sig, t_score = pyperf._utils.is_significant(ref_values, head_values)

            if not sig:
                return None, 0.0
            else:
                ref_values = remove_outliers(ref_values)
                head_values = remove_outliers(head_values)
                values = np.outer(ref_values, 1.0 / head_values).flatten()
                values.sort()
                return values, float(values.mean())

        combined_data = []
        for name, ref in ref_data.items():
            if len(ref) != 0 and name in head_data:
                head = head_data[name]
                if len(ref) == len(head):
                    combined_data.append((name, *calculate_diffs(ref, head)))
        combined_data.sort(key=itemgetter(2))
        return combined_data

    def get_timing_diff(self) -> CombinedData:
        ref_data = self.ref.get_timing_data()
        head_data = self.head.get_timing_data()
        return self._get_combined_data(ref_data, head_data)

    def write_timing_plot(self, filename: Path) -> None:
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

    def write_memory_plot(self, filename: Path) -> None:
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
    def geometric_mean(self) -> str:
        if not self.valid_comparison:
            return ""

        lines = self._contents_lines

        # We want to get the *last* geometric mean in the file, in case
        # it's divided by tags
        for line in lines[::-1]:
            if "Geometric mean" in line:
                geometric_mean = line.split("|")[3].strip()
                break
        else:
            geometric_mean = "not sig"

        return geometric_mean

    def _calculate_memory_change(self):
        # Windows doesn't have memory data
        if self.head.system == "windows":
            return "unknown"

        combined_data = self.get_memory_diff()
        points = np.array([x[2] for x in combined_data if x[1] is not None])
        if len(points) == 0:
            return "unknown"
        else:
            change = np.mean(points)
            return f"{change:.02f}x"

    @functools.cached_property
    def memory_change(self) -> Optional[str]:
        if not self.valid_comparison:
            return ""

        for line in self._contents_lines:
            if line.startswith("- memory change:"):
                return line[line.find(":") + 1 :].strip()

        return None

    @property
    def memory_change_float(self) -> Optional[float]:
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
    def hpt_reliability(self) -> Optional[str]:
        if not self.valid_comparison:
            return ""

        lines = self._contents_lines

        for line in lines:
            m = re.match(r"- Reliability score: (\S+)", line)
            if m is not None:
                return m.group(1)

        return None

    def hpt_percentile(self, percentile: int) -> Optional[str]:
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

    def hpt_percentile_float(self, percentile: int) -> Optional[float]:
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

    def write_pystats_diff(self, filename: Path) -> None:
        try:
            contents = subprocess.check_output(
                [
                    sys.executable,
                    Path("cpython") / "Tools" / "scripts" / "summarize_stats.py",
                    self.ref.filename,
                    self.head.filename,
                ],
                encoding="utf-8",
            )
        except subprocess.CalledProcessError:
            return None
        with open(filename, "w") as fd:
            fd.write(contents)


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
        extra: list[str] = [],
        suffix: str = ".json",
        commit_datetime: Optional[str] = None,
        flags: list[str] = [],
    ):
        self.nickname = nickname
        if nickname not in runners.get_runners_by_nickname():
            raise ValueError(f"Unknown runner {nickname}")
        self.machine = machine
        self.fork = fork
        self.ref = ref
        self.version = version
        self.cpython_hash = cpython_hash
        self.extra = extra
        self.suffix = suffix
        self.flags = sorted(set(flags))
        self._commit_datetime = commit_datetime
        self._filename = None
        self.bases = {}

    @classmethod
    def from_filename(cls, filename: Path) -> "Result":
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
    def from_scratch(
        cls,
        python: Path,
        fork: str,
        ref: str,
        extra: list[str] = [],
        flags: list[str] = [],
    ) -> "Result":
        result = cls(
            _clean(runners.get_nickname_for_hostname(socket.gethostname())),
            _clean(_get_architecture(python)),
            _clean_for_url(fork),
            _clean(ref[:20]),
            _clean(_get_platform_value(python, "python_version")),
            git.get_git_hash(Path("cpython"))[:7],
            extra,
            ".json",
            commit_datetime=git.get_git_commit_date(Path("cpython")),
            flags=flags,
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
    def result_info(self) -> tuple[Optional[str], Optional[str]]:
        match (self.extra, self.suffix):
            case ([], ".json"):
                return ("raw results", None)
            case (["pystats", "vs", base], ".md"):
                return ("pystats diff", base)
            case (["pystats"], ".md"):
                return ("pystats table", None)
            case (["pystats", *_], ".md"):
                return (None, None)
            case (["pystats"], ".json"):
                return ("pystats raw", None)
            case (["vs", base], ".md"):
                return ("table", base)
            case (["vs", base], ".png"):
                return ("time plot", base)
            case (["vs", base, "mem"], ".png"):
                return ("memory plot", base)
        raise ValueError(
            f"Unknown result type (extra={self.extra} suffix={self.suffix})"
        )

    @property
    def fast_contents(self) -> dict[str, Any]:
        """
        Gets just a portion of the JSON contents when the whole set isn't needed.
        """
        if hasattr(self, "_full_contents"):
            return self._full_contents
        if hasattr(self, "_fast_contents"):
            return self._fast_contents

        try:
            import ijson
        except ImportError:
            return self.contents

        def parse_top(parser):
            for prefix, _, _ in parser:
                match prefix:
                    case "benchmarks":
                        parse_benchmarks(parser)
                    case "metadata":
                        parse_metadata(parser)
                    case _:
                        pass

        def parse_benchmarks(parser):
            for prefix, event, value in parser:
                match prefix:
                    case "benchmarks.item.metadata.name":
                        fast_contents["benchmarks"].append(
                            {"metadata": {"name": value}}
                        )
                    case "benchmarks.item.runs.item.metadata.date":
                        if len(fast_contents["benchmarks"]) == 0:
                            fast_contents["benchmarks"].append({})
                        if len(fast_contents["benchmarks"]) == 1:
                            fast_contents["benchmarks"][-1]["runs"] = [
                                {"metadata": {"date": value}}
                            ]
                    case "benchmarks":
                        if event == "end_array":
                            return
                    case _:
                        pass

        def parse_metadata(parser):
            for prefix, event, value in parser:
                if prefix == "metadata" and event == "end_map":
                    return
                elif len(prefix) > 9:
                    fast_contents["metadata"][prefix[9:]] = value

        fast_contents = {"metadata": {}, "benchmarks": []}
        with open(self.filename, "rb") as fd:
            parser = ijson.parse(fd)
            parse_top(parser)

        self._fast_contents = fast_contents
        return fast_contents

    @property
    def contents(self) -> dict[str, Any]:
        if hasattr(self, "_full_contents"):
            return self._full_contents

        if hasattr(self, "_fast_contents"):
            del self._fast_contents

        with open(self.filename, "rb") as fd:
            self._full_contents = ujson.load(fd)

        return self._full_contents

    @property
    def metadata(self) -> dict[str, Any]:
        return self.fast_contents.get("metadata", {})

    @property
    def commit_datetime(self) -> str:
        if self._commit_datetime is not None:
            return self._commit_datetime
        return self.metadata.get("commit_date", "<unknown>")

    @property
    def commit_date(self) -> str:
        return self.commit_datetime[:10]

    @property
    def run_datetime(self) -> str:
        return self.fast_contents["benchmarks"][0]["runs"][0]["metadata"]["date"]

    @property
    def run_date(self) -> str:
        return self.run_datetime[:10]

    @property
    def commit_merge_base(self) -> Optional[str]:
        return self.metadata.get("commit_merge_base", None)

    @property
    def benchmark_hash(self) -> Optional[str]:
        return self.metadata.get("benchmark_hash", None)

    @property
    def hostname(self) -> str:
        return self.metadata.get("hostname", "unknown host")

    @property
    def system(self) -> str:
        return runners.get_runner_by_nickname(self.nickname).os

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
    def github_action_url(self) -> Optional[str]:
        return self.metadata.get("github_action_url", None)

    @property
    def is_tier2(self) -> bool:
        return "PYTHON_UOPS" in self.flags

    @property
    def is_jit(self) -> bool:
        return "JIT" in self.flags

    @property
    def hash_and_flags(self) -> str:
        # A representation for the user that combines the commit hash and other flags
        parts = [self.cpython_hash]
        if self.is_tier2:
            parts.append("️(T2)")
        if self.is_jit:
            parts.append("(JIT)")
        return " ".join(parts)

    @functools.cached_property
    def benchmark_names(self) -> set[str]:
        contents = self.fast_contents
        names = set()
        for benchmark in contents["benchmarks"]:
            if "metadata" in benchmark:
                names.add(benchmark["metadata"]["name"])
            else:
                names.add(contents["metadata"]["name"])
        return names

    @functools.cached_property
    def parsed_version(self):
        from packaging import version as pkg_version

        return pkg_version.parse(self.version.replace("+", "0"))

    def get_timing_data(self) -> dict[str, np.ndarray]:
        data = {}

        for benchmark in self.contents["benchmarks"]:
            name = benchmark.get("metadata", self.contents["metadata"])["name"]
            row = []
            for run in benchmark["runs"]:
                row.extend(run.get("values", []))
            data[name] = np.array(row, dtype=np.float64)

        return data

    def get_memory_data(self) -> dict[str, np.ndarray]:
        data = {}

        # On MacOS, there was a bug in pyperf where the `mem_max_rss` value was
        # erroneously multiplied by 1024.  (BSD defines maxrss in bytes, Linux
        # in kilobytes).

        needs_correction = self.system == "darwin" and version.parse(
            self.contents["metadata"]["perf_version"]
        ) < version.parse("2.6.3")

        for benchmark in self.contents["benchmarks"]:
            name = benchmark.get("metadata", self.contents["metadata"])["name"]
            row = []
            for run in benchmark["runs"]:
                metadata = run.get("metadata", {})
                if mem := metadata.get("command_max_rss"):
                    row.append(mem)
                elif mem := metadata.get("mem_max_rss"):
                    if needs_correction:
                        mem /= 1024
                    row.append(mem)
            data[name] = np.array(row, dtype=np.float64)

        return data


def has_result(
    results_dir: Path,
    commit_hash: str,
    machine: str,
    pystats: bool,
    flags: list[str],
    benchmark_hash: str,
) -> Optional[Result]:
    if machine == "all":
        nickname = None
    else:
        _, _, nickname = machine.split("-")

    results = load_all_results([], results_dir, False)

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


def match_to_bases(results: list[Result], bases: Optional[list[str]]):
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

    groups = defaultdict(lambda: defaultdict(list))
    for result in results:
        if result.fork == "python":
            groups[(result.nickname, tuple(result.extra))][
                result.benchmark_hash
            ].append(result)

    for result in results:
        candidates = groups[(result.nickname, tuple(result.extra))]

        if bases is not None:
            for base in bases:
                find_match(result, candidates, base, lambda ref: ref.version == base)

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
    bases: Optional[list[str]],
    results_dir: Path,
    sorted: bool = True,
    match: bool = True,
) -> list[Result]:
    results = []

    for entry in results_dir.glob("**/*.json"):
        result = Result.from_filename(entry)
        if result.result_info[0] not in ["raw results", "pystats raw"]:
            continue
        results.append(result)
    if len(results) == 0:
        raise ValueError("Didn't find any results.  That seems fishy.")

    if match:
        match_to_bases(results, bases)

    if sorted:
        results.sort(
            key=lambda x: (
                x.parsed_version,
                x.commit_datetime,
            ),
            reverse=True,
        )

    return results
