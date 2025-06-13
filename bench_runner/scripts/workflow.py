from __future__ import annotations


import argparse


import rich_argparse


from bench_runner import flags as mflags


from bench_runner.scripts import benchmark_workflow as mbenchmark_workflow
from bench_runner.scripts import compile as mcompile


def main():
    parser = argparse.ArgumentParser(
        description="""
        Run the full compile/benchmark workflow.
        """,
        formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    )
    mcompile.add_compile_arguments(parser)
    mbenchmark_workflow.add_benchmark_workflow_arguments(parser)
    args = parser.parse_args()

    flags = mflags.parse_flags(args.flags)

    mcompile._main(
        args.fork,
        args.ref,
        args.machine,
        flags,
        args.force,
        args.pgo,
        args.pystats,
        args.force_32bit,
    )
    mbenchmark_workflow._main(
        args.fork,
        args.ref,
        args.benchmarks,
        flags,
        args.perf,
        args.pystats,
        args.force_32bit,
        args.run_id,
        args._fast,
    )


if __name__ == "__main__":
    main()
