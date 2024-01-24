import importlib
import sys

COMMANDS = {
    "backfill": "Schedule benchmarking a number of commits",
    "compare": "Compare a matrix of specific results",
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
}

if __name__ == "__main__":
    command = sys.argv[1]

    if command not in COMMANDS:
        command_length = max(len(k) for k in COMMANDS.keys())
        print(f"Unknown command '{command}'.", file=sys.stderr)
        print(file=sys.stderr)
        print("Valid commands are:", file=sys.stderr)
        for k, v in COMMANDS.items():
            print(f"  {k:{command_length}}: {v}")
        sys.exit(1)

    sys.argv = [sys.argv[0], *sys.argv[2:]]
    mod = importlib.import_module(f"bench_runner.scripts.{command}")
    mod.main()
