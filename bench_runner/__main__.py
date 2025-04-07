COMMANDS = {
    "backfill": "Schedule benchmarking a number of commits",
    "compare": "Compare a matrix of specific results",
    "find_failures": "Find the benchmarks that failed in the last weekly run",
    "generate_results": "Create all of the derived artifacts from raw data",
    "get_merge_base": (
        "Get the merge base of the selected commit, and determine if it should run"
    ),
    "install": "Install the workflow files into a results repository",
    "profiling_plot": "Generate the profiling plots from raw data",
    "purge": "Purge old results from a results repository",
    "remove_benchmark": "Remove specific benchmarks from the data set",
    "run_benchmarks": "Run benchmarks (in timing, pyperf or perf modes)",
    "should_run": "Determine whether we need to rerun results for the current commit",
    "notify": "Send a notification about the completion of the workflow",
}

if __name__ == "__main__":
    import importlib
    import sys

    from .util import rich_print

    command = len(sys.argv) >= 2 and sys.argv[1] or ""

    if command not in COMMANDS:
        command_length = max(len(k) for k in COMMANDS.keys())
        rich_print(f"Unknown command '{command}'.", file=sys.stderr)
        rich_print(file=sys.stderr)
        rich_print("Valid commands are:", file=sys.stderr)
        for k, v in COMMANDS.items():
            rich_print(f"  [blue]{k:{command_length}}[/blue]: {v}")
        sys.exit(1)

    sys.argv = [sys.argv[0], *sys.argv[2:]]
    mod = importlib.import_module(f"bench_runner.scripts.{command}")
    mod.main()
