import argparse
import collections
import errno
import json
import pathlib
import sys
from typing import Iterable

import rich_argparse

# pyperf/pyperformance run benchmarks by benchmark name, but store results,
# including loops used, by data point name (as reported by the benchmark).
# There's no mapping from the one to the other that we can easily use (other
# than running benchmarks one by one and checking what data points they
# report), so here's a hand-written mapping instead. Benchmarks that use
# their own name for the data point are omitted. This will probably (slowly)
# get out of date, but so be it.
#
# (Sorted by value, then key.)
DATAPOINT_TO_BENCHMARK = {
    "many_optionals": "argparse",
    "subparsers": "argparse",
    "async_tree_none": "async_tree",
    "async_tree_none_tg": "async_tree_tg",
    "bench_mp_pool": "concurrent_imap",
    "bench_thread_pool": "concurrent_imap",
    "deepcopy_memo": "deepcopy",
    "deepcopy_reduce": "deepcopy",
    "create_gc_cycles": "gc_collect",
    "genshi_text": "genshi",
    "genshi_xml": "genshi",
    "logging_format": "logging",
    "logging_silent": "logging",
    "logging_simple": "logging",
    "shortest_path": "networkx",
    "connected_components": "networkx_connected_components",
    "k_core": "networkx_k_core",
    "pprint_pformat": "pprint",
    "pprint_safe_repr": "pprint",
    "scimark_fft": "scimark",
    "scimark_lu": "scimark",
    "scimark_monte_carlo": "scimark",
    "scimark_sor": "scimark",
    "scimark_sparse_mat_mult": "scimark",
    "sqlglot_v2_normalize": "sqlglot_v2",
    "sympy_expand": "sympy",
    "sympy_integrate": "sympy",
    "sympy_str": "sympy",
    "sympy_sum": "sympy",
    "xml_etree_generate": "xml_etree",
    "xml_etree_iterparse": "xml_etree",
    "xml_etree_parse": "xml_etree",
    "xml_etree_process": "xml_etree",
}

# The list of bm_* directories in pyperformance and pyston-benchmarks, plus
# the aliases defined in their MANIFEST files (entries with
# '<local:$dirname>')
KNOWN_BENCHMARKS = {
    "2to3",
    "aiohttp",
    "argparse",
    "argparse_subparsers",
    "async_generators",
    "async_tree",
    "async_tree_cpu_io_mixed",
    "async_tree_cpu_io_mixed_tg",
    "async_tree_eager",
    "async_tree_eager_cpu_io_mixed",
    "async_tree_eager_cpu_io_mixed_tg",
    "async_tree_eager_io",
    "async_tree_eager_io_tg",
    "async_tree_eager_memoization",
    "async_tree_eager_memoization_tg",
    "async_tree_eager_tg",
    "async_tree_io",
    "async_tree_io_tg",
    "async_tree_memoization",
    "async_tree_memoization_tg",
    "async_tree_tg",
    "asyncio_tcp",
    "asyncio_tcp_ssl",
    "asyncio_websockets",
    "bpe_tokeniser",
    "chameleon",
    "chaos",
    "comprehensions",
    "concurrent_imap",
    "coroutines",
    "coverage",
    "crypto_pyaes",
    "dask",
    "decimal_factorial",
    "decimal_pi",
    "deepcopy",
    "deltablue",
    "django_template",
    "djangocms",
    "docutils",
    "dulwich_log",
    "fannkuch",
    "flaskblogging",
    "float",
    "gc_collect",
    "gc_traversal",
    "generators",
    "genshi",
    "gevent_hub",
    "go",
    "gunicorn",
    "hexiom",
    "hg_startup",
    "html5lib",
    "json",
    "json_dumps",
    "json_loads",
    "kinto",
    "logging",
    "mako",
    "mdp",
    "meteor_contest",
    "mypy2",
    "nbody",
    "networkx",
    "networkx_connected_components",
    "networkx_k_core",
    "nqueens",
    "pathlib",
    "pickle",
    "pickle_dict",
    "pickle_list",
    "pickle_pure_python",
    "pidigits",
    "pprint",
    "pycparser",
    "pyflate",
    "pylint",
    "python_startup",
    "python_startup_no_site",
    "pytorch_alexnet_inference",
    "raytrace",
    "regex_compile",
    "regex_dna",
    "regex_effbot",
    "regex_v8",
    "richards",
    "richards_super",
    "scimark",
    "spectral_norm",
    "sphinx",
    "sqlalchemy_declarative",
    "sqlalchemy_imperative",
    "sqlglot_v2",
    "sqlglot_v2_optimize",
    "sqlglot_v2_parse",
    "sqlglot_v2_transpile",
    "sqlite_synth",
    "sympy",
    "telco",
    "thrift",
    "tomli_loads",
    "tornado_http",
    "typing_runtime_protocols",
    "unpack_sequence",
    "unpickle",
    "unpickle_list",
    "unpickle_pure_python",
    "xml_etree",
}


def parse_result(results_file, benchmark_data):
    with results_file.open() as f:
        result = json.load(f)
    bms = result["benchmarks"]
    if len(bms) == 1 and "metadata" not in bms[0]:
        # Sometimes a .json file contains just a single benchmark.
        bms = [result]
    for bm in bms:
        if "metadata" not in bm:
            raise RuntimeError(f"Invalid data {bm.keys()!r} in {results_file}")
            return
        name = bm["metadata"]["name"]
        name = DATAPOINT_TO_BENCHMARK.get(name, name)
        assert name is not None  # to satisfy pyright.
        if name not in KNOWN_BENCHMARKS:
            print(
                f"WARNING: unknown benchmark {name!r} in {results_file}",
                file=sys.stderr,
            )
            # Avoid repeated warnings.
            KNOWN_BENCHMARKS.add(name)
        benchmark_data[name].append(bm["metadata"]["loops"])


def _main(
    loops_file: pathlib.Path,
    update: bool,
    overwrite: bool,
    merger: str,
    results: Iterable[pathlib.Path],
):
    if not update and not overwrite and loops_file.exists():
        raise OSError(
            errno.EEXIST,
            f"{loops_file} exists (use -f to overwrite, -u to merge data)",
        )
    benchmark_data = collections.defaultdict(list)
    if update:
        parse_result(loops_file, benchmark_data)
    for result_file in results:
        parse_result(result_file, benchmark_data)

    merge_func = {
        "max": max,
        "min": min,
    }[merger]

    # pyperformance expects a specific layout, and needs the top-level
    # metadata even if it's empty.
    loops_data = {"benchmarks": [], "metadata": {}}
    for bm in sorted(benchmark_data):
        loops = merge_func(benchmark_data[bm])
        bm_result = {"metadata": {"name": bm, "loops": loops}}
        loops_data["benchmarks"].append(bm_result)
    with loops_file.open("w") as f:
        json.dump(loops_data, f, sort_keys=True, indent=4)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="""
        Synthesize a loops.json file for use with `pyperformance`'s
        `--same-loops` (or `PYPERFORMANCE_LOOPS_FILE`) from one or more
        benchmark results.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-o", "--loops_file", help="loops file to write to", required=True
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-u", "--update", action="store_true", help="add to existing loops file"
    )
    group.add_argument(
        "-f", "--overwrite", action="store_true", help="replace loops file"
    )
    parser.add_argument(
        "-s",
        "--select",
        choices=("max", "min"),
        default="max",
        help="how to merge multiple runs",
    )
    parser.add_argument("results", nargs="+", help="benchmark results to parse")
    args = parser.parse_args()

    _main(
        pathlib.Path(args.loops_file),
        args.update,
        args.overwrite,
        args.select,
        [pathlib.Path(r) for r in args.results],
    )


if __name__ == "__main__":
    main()
