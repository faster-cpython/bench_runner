"""
Generate summary tables and a visualization of Linux perf profiling results.
"""

from __future__ import annotations


import argparse
from collections import defaultdict
import csv
import functools
from operator import itemgetter
from pathlib import Path
import re


from matplotlib import pyplot as plt
import numpy as np
import rich_argparse


SANITY_CHECK = True


# Categories of functions, where each value is a list of regular expressions.
# These are matched in-order.
CATEGORIES: dict[str, list[str]] = {
    "interpreter": [
        "_Py_GetBaseOpcode",
        "_PyCode_Quicken",
        "_PyCode_Validate",
        "_PyEval.+",
        "_PyFrame_ClearExceptCode",
        "_PyFrame_New_NoTrack",
        "_PyFrame_Traverse",
        "_PyPegen_.+",
        "_PyThreadState_PopFrame",
        "advance",
        "initialize_locals",
        "intern_string_constants",
        "PyAST_.+",
        "PyEval_.+",
    ],
    "lookup": [
        "_Py_dict_lookup",
        "_Py_type_getattro",
        "_PyType_Lookup",
        "builtin_getattr",
        "find_name_in_mro",
        "lookdict_split",
        "lookdict_unicode_nodummy",
        "lookdict_unicode",
        "PyMember_.*",
        "unicodekeys_lookup_unicode",
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
        "gc_collect_main",
        "gc_collect_region.*",
        "PyObject_IS_GC",
        "type_is_gc",
        "visit_.+",
    ],
    "memory": [
        "_?PyMem_.+",
        "_Py_NewReference",
        "_PyObject_Free",
        "_PyObject_Malloc",
        ".+_alloc",
        ".+[Nn]ew.*",
        ".+Alloc",
        ".+Calloc",
        ".+dealloc",
        ".+Dealloc",
        ".+Realloc",
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
    "library": ["_?sre_.+", "pattern_.+"],
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
        "_?PyDict_.+",
        "build_indices_unicode",
        "clone_combined_dict_keys.+",
        "dict_.+",
        "dictiter_.+",
        "dictresize",
        "find_empty_slot",
        "free_keys_object",
        "insert_to_emptydict",
        "insertdict",
        "new_keys_object",
        "OrderedDict_.+",
    ],
    "list": [
        "_?PyList_.+",
        "list_.+",
        "listiter_.+",
    ],
    "float": [
        "_?PyFloat_.+",
        "float_.+",
    ],
    "str": [
        "_?PyUnicode.+",
        "_copy_characters.+",
        "ascii_decode",
        "bytes_.+",
        "PyBytes_.+",
        "replace",
        "resize_compact",
        "siphash13",
        "split",
        "stringlib_.+",
        "unicode_.+",
    ],
    "miscobj": [
        "_?PySlice_.+",
        "_PyBuildSlice_ConsumeRefs",
        "_PyEval_SliceIndex",
        "bytearray_.+",
        "deque_.+",
        "dequeiter_.+",
        "enum_.+",
        "PyBool_.+",
        "PyBuffer_.+",
        "set_.+",
        "setiter_.+",
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
        "method_vectorcall.+",
        "vectorcall_method",
        "vgetargs1_impl",
        "vgetargskeywords.constprop.0",
    ],
    "import": [
        "PyImport.+",
        "r_.+",
    ],
    "compiler": ["uop_optimize", "_PyJIT_.+"],
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
                if re.match(f"^{pattern}$", sym.split()[0]):
                    return category

    return "unknown"


def _main(input_dir: Path, output_prefix: Path):
    results = defaultdict(lambda: defaultdict(float))
    categories = defaultdict(lambda: defaultdict(float))

    if not input_dir.exists() or len(list(input_dir.glob("*.csv"))) == 0:
        print("No profiling data. Skipping.")
        return

    with output_prefix.with_suffix(".md").open("w") as md:
        for csv_path in sorted(input_dir.glob("*.csv")):
            stem = csv_path.stem.split(".", 1)[0]

            if stem.startswith("sqlalchemy"):
                continue

            md.write(f"\n## {stem}\n\n")
            md.write("| percentage | object | symbol | category |\n")
            md.write("| ---: | :--- | :--- | :--- |\n")

            with csv_path.open(newline="") as fd:
                csvreader = csv.reader(fd)
                for _ in csvreader:
                    break

                # Add up all the JIT entries into a single row
                rows = []
                total = 0.0
                jit_time = 0.0
                for row in csvreader:
                    self_time, _, obj, sym = row
                    self_time = float(self_time)
                    if self_time > 100.0:
                        print(f"{stem} Invalid data")
                    total += self_time
                    if obj == "[JIT]":
                        jit_time += self_time
                    else:
                        rows.append((self_time, obj, sym))
                if jit_time != 0.0:
                    rows.append((jit_time, "[JIT]", "jit"))

                if total >= 100.0:
                    rows = [
                        (self_time / 100.0, obj, sym) for (self_time, obj, sym) in rows
                    ]

                rows.sort(reverse=True)

                for self_time, obj, sym in rows:
                    # python3.8 is the "parent" python orchestrating pyperformance
                    if obj == "python3.8":
                        continue

                    if self_time <= 0.0:
                        break

                    category = category_for_obj_sym(obj, sym)
                    categories[category][(obj, sym)] += self_time

                    results[stem][category] += self_time

                    if self_time >= 0.005:
                        md.write(
                            f"| {self_time:.2%} | `{obj}` | `{sym}` | {category} |\n"
                        )

        sorted_categories = sorted(
            [
                (sum(val.values()) / len(results), key)
                for (key, val) in categories.items()
            ],
            reverse=True,
        )

        md.write("\n\n## Categories\n")
        for total, category in sorted_categories:
            matches = categories[category]
            md.write(f"\n### {category}\n\n")
            md.write(f"{total:.2%} total\n\n")
            md.write("| percentage | object | symbol |\n")
            md.write("| ---: | :--- | :--- |\n")
            for (obj, sym), self_time in sorted(
                matches.items(), key=itemgetter(1), reverse=True
            ):
                if self_time < 0.005:
                    break
                md.write(f"| {self_time / len(results):.2%} | {obj} | {sym} |\n")

    fig, ax = plt.subplots(figsize=(8, len(results) * 0.3), layout="constrained")

    bottom = np.zeros(len(results))
    names = list(results.keys())[::-1]

    for val, category in sorted_categories:
        if category == "unknown":
            continue
        values = np.array(
            [results[name].get(category, 0.0) for name in names], np.float64
        )
        color, hatch = get_color_and_hatch(category)
        ax.barh(
            names,
            values,
            0.5,
            label=f"{category} {val:.2%}",
            left=bottom,
            hatch=hatch,
            color=color,
        )
        bottom += values

    values = 1.0 - bottom
    ax.barh(names, values, 0.5, label="(other functions)", left=bottom, color="#ddd")

    ax.set_xlabel("percentage time")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    ax.set_xlim([0, 1])

    fig.savefig(output_prefix.with_suffix(".svg"))

    fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
    values = [x[0] for x in sorted_categories]
    labels = [
        i < 10 and f"{x[1]} {x[0]:.2%}" or "" for i, x in enumerate(sorted_categories)
    ]
    colors = [get_color_and_hatch(cat[1])[0] for cat in sorted_categories]
    hatches = [get_color_and_hatch(cat[1])[1] for cat in sorted_categories]

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

    fig.savefig(output_prefix.with_suffix(".pie.svg"), dpi=200)


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
