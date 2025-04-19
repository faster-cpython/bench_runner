## Unreleased

### Moving more code to Python

Most of the code to orchestrate the benchmarks at a high level has moved from
GitHub Actions `.yml` files to Python code in `workflow.py`. In the future, this
will allow supporting more workflow engines beyond just GitHub Actions.

**Migration note**: After running `python -m bench_runner install` to update
your local files, be sure to add the new `workflow_bootstrap.py` file to your
git repository.

### Decoupling compiler configuration from tail-calling

Previously, bench_runner had a flag, `CLANG`, that both (a) built with clang 19
or later and (b) built the tail calling interpreter. This has been replaced with
a new flag `TAILCALL` that only sets the flags to build with the tail calling
interpreter. It is now up to the user to select a machine configured with a
compiler that supports the tail calling interpreter.

For machines with a clang-19 or later installed these example machine
configurations could be used in your `bench_runner.toml`. The exact details will
depend on your distribution and method of installing clang, etc.:

```toml
[runners.linux_clang]
os = "linux"
arch = "x86_64"
hostname = "pyperf"
github_runner_name = "linux-x86_64-linux"
include_in_all = false
[runners.linux_clang.env]
CC = "$(which clang-19)"
LLVM_AR = "$(which llvm-ar-19)"
LLVM_PROFDATA = "$(which llvm-profdata-19)"

[runners.darwin_clang19]
os = "darwin"
arch = "arm64"
hostname = "CPythons-Mac-mini.local"
github_runner_name = "darwin-arm64-darwin"
[runners.darwin_clang19.env]
PATH = "$(brew --prefix llvm)/bin:$PATH"
CC = "$(brew --prefix llvm)/bin/clang"
LDFLAGS = "-L$(brew --prefix llvm)/lib"
CFLAGS = "-L$(brew --prefix llvm)/include"

[runners.pythonperf1_clang]
os = "windows"
arch = "x86_64"
hostname = "WIN-5FKPU9U7KDT"
github_runner_name = "windows-x86_64-pythonperf1"
include_in_all = false
[runners.pythonperf1_clang.env]
BUILD_DEST = "PCBuild/amd64"
PYTHON_CONFIGURE_FLAGS = '`"/p:PlatformToolset=clangcl`" `"/p:LLVMInstallDir=C:\Program Files\LLVM`" `"/p:LLVMToolsVersion=19.1.6`"'
```

### Controlling number of cores used for builds

Runners have a new configuration `use_cores` to control the number of CPU cores
used to build CPython. By default, this will place no limit on the number of
cores used, but machines with limited RAM may require using fewer cores.

### Configurable weekly runs

The set of weekly runs are now configured from the `bench_runner.toml`.
If you don't add any configuration, the only weekly runs performed will be default builds on every runner.

The `weekly` section in the configuration is made up of sections, each of which
has a `flags` parameter specifying the flags to use and a `runner` parameter
specifying the runners to run on.

For example:

```toml
[weekly.default]
flags = []
runners = ["linux", "darwin", "windows"]

[weekly.tailcall]
flags = ["TAILCALL"]
runners = ["linux_clang", "darwin", "windows_clang"]
```

### New plot configuration

The plot configuration has been completely overhauled.

If you have a `[plot]` section in `bench_runner.toml` it should be removed.
Runner-specific plot configuration is now in the `plot` key in each runner table.
Configuration of the longitudinal and flag effect plots are in the new `[longitudinal_plot]` and `[flag_effect_plot]` sections.
See `README.md` for more information.

### Long-running tests

`bench-runner` has some new long-running end-to-end integration tests. To avoid running them, use:

```
python -m pytest -m "not long_running"
```

### Configurable groups

Runners can configure which groups they belong to, and benchmark runs can be
started on whole groups. Unlike when using 'all' or individual runners, runs
for which the result already exists are skipped. For example, a
configuration like:

```toml
[runners.linux_clang]
os = "linux"
arch = "x86_64"
hostname = "pyperf1"
env.CC = "clang"
groups = ["linux", "pyperf1", "clang"]

[runners.linux_gcc]
os = "linux"
arch = "x86_64"
hostname = "pyperf1"
groups = ["linux", "pyperf1", "gcc"]

[runners.linux2_gcc]
os = "linux"
arch = "x86_64"
hostname = "pyperf2"
groups = ["linux", "pyperf2", "gcc"]
```

... will add `group linux`, `group pyperf1`, `group pyperf2`, `group gcc`
and `group clang` to the list of possible machines for benchmark runs.
Selecting `group linux` will queue a run for all three runners, and `group
pyperf1` only for the first two.

## v1.8.0

### bench_runner.toml change

The configuration no longer expects a top-level, single element `runners` array.

You should remove the following line from your `bench_runners.toml`:

```toml
[[runners]]
```
