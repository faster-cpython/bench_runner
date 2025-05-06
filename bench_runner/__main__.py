import importlib
import os
import sys


import rich


COMMANDS = {
    "backfill": "Schedule benchmarking a number of commits",
    "bisect": "Run a bisect to find the commit that caused a regression",
    "compare": "Compare a matrix of specific results",
    "find_failures": "Find the benchmarks that failed in the last weekly run",
    "generate_results": "Create all of the derived artifacts from raw data",
    "get_merge_base": (
        "Get the merge base of the selected commit, and determine if it should run"
    ),
    "install": "Install the workflow files into a results repository",
    "notify": "Send a notification about the completion of the workflow",
    "profiling_plot": "Generate the profiling plots from raw data",
    "purge": "Purge old results from a results repository",
    "remove_benchmark": "Remove specific benchmarks from the data set",
    "run_benchmarks": "Run benchmarks (in timing, pyperf or perf modes)",
    "synthesize_loops_file": "Create a loops file from multiple benchmark results",
    "workflow": "Run the full compile/benchmark workflow",
}

if __name__ == "__main__":
    # This lets pytest-cov collect coverage data in a subprocess
    if "COV_CORE_SOURCE" in os.environ:
        try:
            from pytest_cov.embed import init

            init()
        except Exception:
            sys.stderr.write("pytest-cov: Failed to setup subprocess coverage.")

    command = len(sys.argv) >= 2 and sys.argv[1] or ""

    if command not in COMMANDS:
        command_length = max(len(k) for k in COMMANDS.keys())
        rich.print(f"Unknown command '{command}'.", file=sys.stderr)
        rich.print(file=sys.stderr)
        rich.print("Valid commands are:", file=sys.stderr)
        for k, v in COMMANDS.items():
            rich.print(f"  [blue]{k:{command_length}}[/blue]: {v}")
        sys.exit(1)

    sys.argv = [sys.argv[0], *sys.argv[2:]]
    mod = importlib.import_module(f".{command}", "bench_runner.scripts")
    mod.main()
