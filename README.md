# Benchmarking tools for the Faster CPython project

## Usage

This code lets you set up your own Github repo to run pyperformance benchmarks on your own self-hosted Github Action runners.

For example, you can see [the Faster CPython team's benchmarking results](https://github.com/faster-cpython/benchmarking-public).

### Set up the repo

Create a new empty repository on Github and clone it locally.

Add bench_runner to your `requirements.txt`.

```text
bench_runner=={VERSION}
```

Replace the {VERSION} above with the latest version tag of `bench_runner`.

Create a virtual environment and install your requirements to it, for example:

```bash session
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### Generate workflows

Run the install script to generate the files to make the Github Actions work (from the root of your repo):

```bash session
python -m bench_runner install
```

This will create some files in `.github/workflows` as well as some configuration files at the root of your repo.
Commit them to your repository, and push up to Github.

```bash session
git commit -a -m "Initial commit"
git push origin main
```

The `bench_runner.toml` file created at the root of your repository contains configuration specific to your instance.
More details about this configuration are below.
Every time you make a change to the `bench_runner.toml` file, you will need to rerun `python -m bench_runner install` to have the changes reflected.

### Add some self-hosted runners

Provision the machine to have the build requirements for CPython and the base requirements for Github Actions according to the [provisioning instructions](PROVISIONING.md).

Then, add it to the pool of runners by following the instructions on Github's `Settings -> Actions -> Runners -> Add New Runner` to add a new runner.

The default responses to all questions should be fine _except_ pay careful attention to set the labels correctly.
Each runner must have the following labels:

- One of `linux`, `macos` or `windows`.
- `bare-metal` (to distinguish it from VMs in the cloud).
- `$os-$arch-$nickname`, where:
  - `$os` is one of `linux`, `macos`, `windows`
  - `$arch` is one of `x86_64` or `arm64` (others may be supported in future)
  - `$nickname` is a unique nickname for the runner.

Once the runner is set up, [enable it as a service](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/configuring-the-self-hosted-runner-application-as-a-service) so it will start automatically on boot.

In addition, the metadata about the runner(s) must be added to `runners` section in `bench_runner.toml`, for example:

```toml
[runners.linux]
os = "linux"
arch = "x86_64"
hostname = "pyperf"
```

You may also want to add a specific machine to collect pystats.
Since this machine doesn't need to run on bare metal to have accurate timings, it usually is a cloud instance.
Give this machine the special label `cloud` to indicate it is available for collecting pystats.
Additionally, flag it as `available = false` in its configuration so it won't be used to collect timings.

```
[runners.pystats]
os = "linux"
arch = "x86_64"
hostname = "cpython-benchmarking-azure"
available = false
```

If you don't want a machine to be included when the user selects "machine == 'all'", add:

```
include_in_all = false
```

You may limit the number of cores used to build Python with the `use_cores` option. This may be necessary, for example, on cloud VMs with limited RAM.

```
use_cores = 2
```

### Try a benchmarking run

There are instructions for running a benchmarking action already in the `README.md` of your repo. Look there and give it a try!

### Additional configuration

#### Set of benchmarks

By default, all of the benchmarks in `pyperformance` and `python-macrobenchmarks` are run. To configure the set of benchmarks, or add more, edit the `benchmarks.manifest` file.
The format of this file is documented with `pyperformance`.

You can also exclude specific benchmarks by adding them to the `benchmarks/excluded_benchmarks` value in your `bench_runner.toml` file.

#### Reference versions

All benchmarked commits are automatically compared to key "reference" versions, as well as their merge base, if available.
The reference versions are defined in the `bases/versions` value in your `bench_runner.toml` file.
Don't forget to actually collect benchmark data for those tags -- it's doesn't happen automatically.

By default, pyperformance will determine the number of times to run each benchmark dynamically at runtime, by choosing a number at which the timing
measurement becomes stable.
However, this can make comparing benchmark runs less accurate.
It is recommended to specify one of your base benchmarking runs as the source of a hardcoded number of loops.
To do so, add a symlink called `loops.json` to the root of your repository that points to a baseline benchmarking run, for example:

```sh
ln -s results/bm-20231002-3.12.0-0fb18b0/bm-20231002-linux-x86_64-python-v3.12.0-3.12.0-0fb18b0.json loops.json
```

#### Plot configuration

`bench_runner` will produce longitudinal plots comparing versions in a series to a specific base version, as well as showing the effect of various flags on the same commits over time.

##### Runner plot styles

For each runner in your `bench_runner.toml`, you can specify a `plot` table with the following keys to control how that runner is rendered in the longitudinal plots:

- `name`: A human-friendly name to display in the plot legend
- `style`: A [matplotlib line style](https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle)
- `marker`: A [matplotlib marker](https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers)
- `color`: A [matplotlib color](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def)

##### Longitudinal plot configuration

The longitudinal plot shows the change of a version branch over time against a specified base version. It is made up of multiple subplots, each with its own head and base, and optionally configuration flags.

In `bench_runner.toml`, the `longitudinal_plot` table has a `subplots` key which is an array of tables with the following keys:

- `base`: The base version to compare to. Should be a fully-specified version, e.g. "3.13.0".
- `version`: The version series to use as a head. Should be a 2-part version, e.g. "3.14"
- `flags`: (optional) A list of flags to match to for the head versions
- `runners`: (optional) A list of nicknames of runners to plot. Defaults to all runners.

For example:

```toml
[longitudinal_plot]
subplots = [
    { base = "3.10.4", version = "3.11" },
    { base = "3.12.0", version = "3.13" },
    { base = "3.13.0", version = "3.14", runners = ["linux1", "linux2"] },
    { base = "3.13.0", version = "3.14", runners = ["windows1", "macos1"] },
    { base = "3.13.0", version = "3.14", flags = ["JIT"] }
]
```

##### Flag effect plot configuration

The flag effect plot shows the effect of specified configuration flags against a base with the same commit hash, but different configuration flags.

In `bench_runner.toml`, the `flag_effect_plot` table has a `subplots` key which is an array of tables with the following keys:

- `name`: The description of the flags to use in the title.
- `version`: The version series to compare. Should be a 2-part version, e.g. "3.14"
- `head_flags`: A list of flags to use as the head.
- `base_flags`: (optional) A list of flags to use as the base. By default, this is a default build, i.e. no flags.
- `runners`: (optional) A list of nicknames of runners to plot. Defaults to all runners.
- `runner_map`: (optional) If you need to map a runner to a base in a
  different runner, you can provide that mapping here. For example, with
  tail-calling, you may want to compare runners configured to use clang
  against runners configured with the "default compiler" for a given
  platform. The mapping is from the "head" runner nickname to the "base"
  runner nickname. If `runner_map` is not empty, only the "head" runners in
  the map are plotted.

For example:

```toml
[[flag_effect_plot.subplots]]
name = "JIT"
version = "3.14"
head_flags = ["JIT"]

[[flag_effect_plot.subplots]]
name = "Tail calling interpreter"
version = "3.14"
head_flags = ["TAILCALL"]
runners = ["linux_clang"]
runner_map = { linux_clang = "linux" }
```

##### Benchmark longitudinal plot configuration

The benchmark longitudinal plot shows the change over time, per benchmark. The configuration consists of the following keys:

- `base`: The base version
- `version`: The version to track
- `runners`: The runners to show
- `head_flags`: (optional) The flags to use for the head commits
- `base_flags`: (optional) The flags to use for the base commits

#### Purging old data

With a local checkout of your results repository you can perform some maintenance tasks.

Clone your results repository, and then install the correct version of bench_runner into a virtual environment to use the bench_runner command:

```
git clone {YOUR_RESULTS_REPO}
cd {YOUR_RESULTS_REPO}
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Periodically, you will want to run the `purge` command to remove old results that aren't of a tagged Python release.

```
python -m bench_runner purge
```

To see more options that control what is deleted, run `python -m bench_runner purge --help`.

After purging the results, you will usually want to squash the git history down to a single commit to save space in your repository. **NOTE THAT THIS IS A DESTRUCTIVE OPERATION THAT WILL DELETE OLD DATA.**

```
git checkout --orphan new-main main
git commit -m "Purging old results on ..."

# Overwrite the old master branch reference with the new one
git branch -M new-main main
git push -f origin main
```

### Running

## Developer

To learn how to hack on this project, see the full [developer documentation](DEVELOPER.md).
