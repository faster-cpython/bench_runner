## Unreleased

## v2.0.0

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
used to build CPython. By default, this will use all available cores, but some
Cloud VMs require using fewer.
