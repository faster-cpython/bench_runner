## Unreleased

## v2.0.0

Most of the work has moved from GitHub Actions `.yml` files to Python code in `workflow.py`.
In the future, this will allow supporting more workflow engines beyond just GitHub Actions.

**Migration note**: After running `python -m bench_runner install` to update your local files, but sure to add the new `workflow_bootstrap.py` file to your git repository.

### New configuration

Runners have a new configuration `use_cores` to control the number of CPU cores
used to build CPython. By default, this will use all available cores, but some
Cloud VMs require using fewer.
