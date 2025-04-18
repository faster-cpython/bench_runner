## Unreleased

### bench_runner.toml change

The configuration no longer expects a top-level, single element `runners` array.

You should remove the following line from your `bench_runners.toml`:

```
[[runners]]
```
