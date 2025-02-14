"""
Generate summary tables and a visualization of Linux perf profiling results.
"""

from __future__ import annotations


import argparse
from collections import defaultdict
import csv
import functools
import json
from operator import itemgetter
from pathlib import Path
import re
from typing import IO


from matplotlib import pyplot as plt
import numpy as np
import rich_argparse


from bench_runner.util import PathLike


SANITY_CHECK = True

# Must match the value in _benchmark.src.yml
PERF_PERIOD = 1000000


# Categories of functions, where each value is a list of regular expressions.
# These are matched in-order.
CATEGORIES: dict[str, list[str]] = {
    "interpreter": [
        "_Py_call_instrumentation_line",
        "_Py_GetBaseOpcode",
        "_PyCode_.+",
        "_PyEval.+",
        "_PyFrame_.+",
        "_PyPegen_.+",
        "_PyStack_.+",
        "_PyVectorcall_.+",
        "_TAIL_CALL_.+",
        "advance",
        "call_instrumentation_vector.*",
        "initialize_locals",
        "PyAST_.+",
        "PyEval_.+",
        "PyInterpreterState.*",
    ],
    "lookup": [
        "_Py_dict_lookup.*",
        "_Py_hashtable_.+",
        "_Py_type_getattro",
        "_PyType_Lookup",
        "_PyType_LookupRef.*",
        "builtin_getattr",
        "find_name_in_mro",
        "lookdict_split",
        "lookdict_unicode_nodummy",
        "lookdict_unicode",
        "lookup_maybe_method",
        "member_get",
        "PyMember_.*",
        "unicodekeys_lookup_unicode.*",
        "update_one_slot",
    ],
    "gc": [
        "_?PyGC_.+",
        "_?PyObject_GC_.+",
        "_PyObject_Visit.+",
        "_PyTrash_.+",
        ".+_traverse",
        ".+MaybeUntrack",
        "deduce_unreachable",
        "gc_collect.*",
        "mark_heap_visitor",
        "mark_stacks",
        "PyObject_IS_GC",
        "scan_heap.+",
        "type_is_gc",
        "update_refs.*",
        "visit_.+",
    ],
    "memory": [
        "_?mi_.+",
        "_?PyMem_.+",
        "_Py_DecRefShared",
        "_Py_MergeZeroLocalRefcount",
        "_Py_NewReference",
        "_py_qsbr_.+",
        "_Py_TryIncrefCompare",
        "_PyObject_Free",
        "_PyObject_Malloc",
        "_PyType_AllocNoTrack",
        ".+_alloc",
        ".+[Nn]ew.*",
        ".+Alloc",
        ".+Calloc",
        ".+dealloc",
        ".+Dealloc",
        ".+MiMalloc",
        ".+Realloc",
        "memcpy.+",
        "memset.+",
        "Py_DECREF",
    ],
    "dynamic": [
        "_?PyMapping_.+",
        "_?PyNumber_.+",
        "_?PyObject_.+",
        "_?PySequence_.+",
        "_Py_type_getattro_impl",
        "_PySuper_Lookup",
        "_PyType_GetDict",
        "delitem_common",
        "do_super_lookup",
        "getset_get",
        "method_get",
        "object_.+",
        "PyDescr_.+",
        "PyIter_.+",
        "PyType_GetModuleByDef",
        "PyType_IsSubtype",
        "slot_tp_richcompare",
        "StopIteration.+",
        "type_.+",
    ],
    "library": ["_?sre_.+", "pattern_.+", "convertitem.+", "sys_trace_.+"],
    "int": [
        "_?PyLong_.+",
        "Balloc",
        "k_.+",
        "l_.+",
        "long_.+",
        "x_.+",
    ],
    "tuple": [
        "_?PyTuple_.+",
        "tuple.+",
    ],
    "dict": [
        "_?PyDict.+",
        "acquire_key_value",
        "build_indices_unicode",
        "clone_combined_dict_keys.+",
        "dict_.+",
        "dictiter_.+",
        "dictkeys_.+",
        "dictresize",
        "find_empty_slot",
        "free_keys_object",
        "insert_combined_dict",
        "insert_to_emptydict",
        "insertdict",
        "new_dict",
        "new_keys_object",
        "OrderedDict_.+",
    ],
    "list": [
        "_?PyList_.+",
        "_?list_.+",
        "listiter_.+",
    ],
    "float": [
        "_?PyFloat_.+",
        "float_.+",
    ],
    "str": [
        "_?PyUnicode.+",
        "_copy_characters.*",
        "ascii_decode",
        "bytes_.+",
        "find_first_nonascii",
        "intern_common.*",
        "intern_constants",
        "intern_string_constants",
        "PyBytes_.+",
        "replace",
        "resize_compact",
        "siphash13",
        "split",
        "stringlib_.+",
        "unicode_.+",
    ],
    "miscobj": [
        "_?PyGen_.+",
        "_?PySet_.+",
        "_?PySlice_.+",
        "_PyBuildSlice_ConsumeRefs",
        "_PyEval_SliceIndex",
        "_PyMake_Coro",
        "bytearray_.+",
        "deque_.+",
        "dequeiter_.+",
        "enum_.+",
        "gen_iternext",
        "get_or_create_weakref",
        "make_gen",
        "PyBool_.+",
        "PyBuffer_.+",
        "range_.+",
        "set_.+",
        "setiter_.+",
        "weakref_.+",
    ],
    "exceptions": [
        "_?PyErr_.*",
        ".+Error_init",
        "BaseException.*",
        "PyCode_Addr2Line",
        "PyException_.*",
        "PyFrame_.*",
        "PyTraceBack_.+",
    ],
    "gil": [
        "drop_gil",
        "PyGILState_.*",
        "take_gil",
    ],
    "calls": [
        "_?PyArg_.+",
        "_Py_CheckFunctionResult",
        "_PyFunction_Vectorcall",
        "cfunction_call.*",
        "cfunction_vectorcall.+",
        "method_vectorcall.*",
        "vectorcall_method",
        "vgetargs1_impl",
        "vgetargskeywords.constprop.0",
    ],
    "import": [
        "PyImport.+",
        "r_.+",
    ],
    "compiler": [
        "_Py_uop_analyse_and_optimize",
        "optimize_uops",
        "uop_optimize",
        "_PyJIT_.+",
        "tok_.+",
    ],
    "async": ["async_.+"],
    "threading": [
        "_?PyThreadState_.+",
        "_PyCriticalSection.+",
        "_PySeqLock_.+",
        "disable_deferred_refcounting",
        "PyThread.*",
    ],
}

COLOR_ORDER = ["jit", "kernel", "libc", "library"] + list(CATEGORIES.keys())


def get_color_and_hatch(category: str) -> tuple[str, str]:
    hatches = ["", "//", "\\\\"]
    try:
        index = COLOR_ORDER.index(category)
    except ValueError:
        return "#ddd", ""

    return f"C{index % 10}", hatches[index // 10]


@functools.cache
def category_for_obj_sym(obj: str, sym: str) -> str:
    if obj == "[kernel.kallsyms]":
        return "kernel"

    if obj.startswith("libc"):
        return "libc"

    if obj == "[JIT]":
        return "jit"

    if re.match(r".+\.so(\..+)?$", obj):
        return "library"

    if obj == "python":
        for category, patterns in CATEGORIES.items():
            for pattern in patterns:
                if re.match(f"^{pattern}$", sym.split()[0].split(".")[0]):
                    return category

    return "unknown"


def handle_benchmark(
    csv_path: PathLike,
    md: IO[str],
    results: defaultdict[str, defaultdict[str, float]],
    categories: defaultdict[str, defaultdict[tuple[str, str], float]],
) -> float:
    csv_path = Path(csv_path)

    stem = csv_path.stem.split(".", 1)[0]

    md.write(f"\n## {stem}\n\n")
    md.write("| percentage | object | symbol | category |\n")
    md.write("| ---: | :--- | :--- | :--- |\n")

    # `python` processes that have symbols coming from a shared object
    # called pythonX.XX or libpythonX.XX.so are from the orchestrating
    # Python, not the one under benchmarking, so they should be skipped.
    tainted_pids = set()
    with csv_path.open(newline="") as fd:
        csvreader = csv.reader(fd)
        for _ in csvreader:
            break

        for self_time, pid, _, obj, _ in csvreader:
            if re.match(r"python[0-9]+\.[0-9]+", obj) or re.match(
                r"libpython[0-9]+\.[0-9]+\.so", obj
            ):
                tainted_pids.add(pid)

    times = defaultdict(float)
    total = 0.0
    with csv_path.open(newline="") as fd:
        csvreader = csv.reader(fd)
        for _ in csvreader:
            break

        for self_time, pid, command, obj, sym in csvreader:
            if pid in tainted_pids:
                continue

            if command != "python":
                continue

            self_time = float(self_time)
            if obj == "[JIT]":
                times[("[JIT]", "jit")] += self_time
            else:
                times[(obj, sym)] += self_time
            total += self_time

    scale = 1.0 / total
    rows = sorted(((v, *k) for k, v in times.items()), reverse=True)

    for self_time, obj, sym in rows:
        if self_time <= 0.0:
            break

        category = category_for_obj_sym(obj, sym)
        categories[category][(obj, sym)] += self_time

        results[stem][category] += self_time

        scaled_time = self_time * scale
        if scaled_time >= 0.0025:
            md.write(f"| {scaled_time:.2%} | `{obj}` | `{sym}` | {category} |\n")

    return total


def plot_bargraph(
    results: defaultdict[str, defaultdict[str, float]],
    categories: list[tuple[float, str]],
    output_filename: PathLike,
):
    fig, ax = plt.subplots(figsize=(8, len(results) * 0.3), layout="constrained")

    bottom = np.zeros(len(results))
    names = list(results.keys())[::-1]
    dens = {key: sum(val.values()) for key, val in results.items()}
    den = sum(x[0] for x in categories)

    for val, category in categories:
        if category == "unknown":
            continue
        values = np.array(
            [results[name].get(category, 0.0) / dens[name] for name in names],
            np.float64,
        )
        color, hatch = get_color_and_hatch(category)
        ax.barh(
            names,
            values,
            0.5,
            label=f"{category} {val / den:.2%}",
            left=bottom,
            hatch=hatch,
            color=color,
        )
        bottom += values

    values = 1.0 - bottom
    ax.barh(names, values, 0.5, label="(other functions)", left=bottom, color="#ddd")

    ax.set_xlabel("percentage time")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    ax.set_xlim((0, 1))

    fig.savefig(output_filename)


def plot_pie(categories: list[tuple[float, str]], output_filename: PathLike):
    fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
    values = [x[0] for x in categories]
    den = sum(values)
    labels = [
        i < 10 and f"{x[1]} {x[0] / den:.2%}" or "" for i, x in enumerate(categories)
    ]
    colors = [get_color_and_hatch(cat[1])[0] for cat in categories]
    hatches = [get_color_and_hatch(cat[1])[1] for cat in categories]

    if sum(values) < 1.0:
        other = 1.0 - sum(values)
    else:
        other = 0.0
    values.append(other)
    labels.append("")
    colors.append("#ddd")
    hatches.append("")

    ax.pie(
        values, labels=labels, colors=colors, hatch=hatches, textprops={"fontsize": 6}
    )

    fig.savefig(output_filename, dpi=200)


def handle_tail_call_stats(
    input_dir: PathLike,
    categories: defaultdict[str, defaultdict[tuple[str, str], float]],
    output_prefix: PathLike,
):
    input_dir = Path(input_dir)
    output_prefix = Path(output_prefix)

    tail_call_stats = defaultdict(float)
    total_time = 0.0
    for (_, sym), self_time in categories["interpreter"].items():
        if (bytecode := sym.removeprefix("_TAIL_CALL_")) != sym:
            tail_call_stats[bytecode] += self_time
            total_time += self_time

    if len(tail_call_stats) == 0:
        return

    pystats_file = input_dir / "pystats.json"

    if not pystats_file.is_file():
        print("No pystats.json file found. Skipping tail call stats.")
        return

    with pystats_file.open() as fd:
        pystats = json.load(fd)

    pystats_bytecodes = defaultdict(int)
    total_count = 0
    for key, val in pystats.items():
        if match := re.match(r"opcode\[(.+)\]\.execution_count", key):
            pystats_bytecodes[match.group(1)] += val
            total_count += val

    with open(output_prefix.with_suffix(".tail_calls.csv"), "w") as csvfile:
        writer = csv.writer(csvfile, dialect="unix")
        writer.writerow(
            ["Bytecode", "% time", "count", "% count", "time per count (μs)"]
        )
        for bytecode, periods in sorted(
            tail_call_stats.items(), key=itemgetter(1), reverse=True
        ):
            count = pystats_bytecodes[bytecode]
            if count == 0:
                continue
            writer.writerow(
                [
                    bytecode,
                    f"{periods / total_time:.02%}",
                    count,
                    f"{count / total_count:.02%}",
                    f"{((periods / PERF_PERIOD) / count) * 1e6:03f}",
                ]
            )


def _main(input_dir: PathLike, output_prefix: PathLike):
    input_dir = Path(input_dir)
    output_prefix = Path(output_prefix)

    results = defaultdict(lambda: defaultdict(float))
    categories = defaultdict(lambda: defaultdict(float))

    if not input_dir.exists() or len(list(input_dir.glob("*.csv"))) == 0:
        print("No profiling data. Skipping.")
        return

    total = 0.0
    with output_prefix.with_suffix(".md").open("w") as md:
        for csv_path in sorted(input_dir.glob("*.csv")):
            if ".tail_calls.csv" in csv_path.name:
                continue
            total += handle_benchmark(csv_path, md, results, categories)

        sorted_categories = sorted(
            [(sum(val.values()), key) for (key, val) in categories.items()],
            reverse=True,
        )

        md.write("\n\n## Categories\n")
        for category_total, category in sorted_categories:
            matches = categories[category]
            md.write(f"\n### {category}\n\n")
            md.write(f"{category_total / total:.2%} total\n\n")
            md.write("| percentage | object | symbol |\n")
            md.write("| ---: | :--- | :--- |\n")
            for (obj, sym), self_time in sorted(
                matches.items(), key=itemgetter(1), reverse=True
            ):
                self_fraction = self_time / total
                if self_fraction < 0.000025:
                    break
                md.write(f"| {self_fraction:.2%} | {obj} | {sym} |\n")

    plot_bargraph(results, sorted_categories, output_prefix.with_suffix(".svg"))
    plot_pie(sorted_categories, output_prefix.with_suffix(".pie.svg"))

    handle_tail_call_stats(input_dir, categories, output_prefix)


def main():
    parser = argparse.ArgumentParser(
        description="Generate graphs from profiling information",
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        default=Path(),
        help="The location of the .csv files of profiling data",
    )
    parser.add_argument(
        "output",
        type=Path,
        default=Path(),
        help="The path and file prefix for the output files",
    )

    args = parser.parse_args()

    _main(args.input_dir, args.output)


if __name__ == "__main__":
    main()
