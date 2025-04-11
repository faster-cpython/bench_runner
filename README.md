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

In addition, the metadata about the runner must be added to `runners` section in `bench_runner.toml`, for example:

```toml
[[runners]]
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

You may limit the number of cores used to build Python with the `use_cores` option. This may be necessary, for example, on cloud VMs.

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

#### Longitudinal plots

The longitudinal plots are configured in the `plot` section of `bench_runner.toml`.

**TODO: Describe this in more detail**

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
