# Faster CPython Benchmark Infrastructure

🔒 [▶️ START A BENCHMARK RUN](https://github.com/faster-cpython/benchmarking/actions/workflows/benchmark.yml)

For documentation about how this works, see the [developer docs](DEVELOPER.md).

## Running benchmarks from the GitHub web UI

Visit the 🔒 [benchmark action](https://github.com/faster-cpython/benchmarking/actions/workflows/benchmark.yml) and click the "Run Workflow" button.

The available parameters are:

- `fork`: The fork of CPython to benchmark.
  If benchmarking a pull request, this would normally be your GitHub username.
- `ref`: The branch, tag or commit SHA to benchmark.
  If a SHA, it must be the full SHA, since finding it by a prefix is not supported.
- `machine`: The machine to run on.
  One of `linux-amd64` (default), `windows-amd64`, `darwin-arm64` or `all`.
- `benchmark_base`: If checked, the base of the selected branch will also be benchmarked.
  The base is determined by running `git merge-base upstream/main $ref`.
- `pystats`: If checked, collect the pystats from running the benchmarks.
- `publish`: If checked, the results will be published in the public [ideas repo](https://github.com/faster-cpython/ideas) upon successful completion.

To watch the progress of the benchmark, select it from the 🔒 [benchmark action page](https://github.com/faster-cpython/benchmarking/actions/workflows/benchmark.yml).
It may be canceled from there as well.
To show only your benchmark workflows, select your GitHub ID from the "Actor" dropdown.

When the benchmarking is complete, the results are published to this repository and will appear in the [master table](results/README.md).
Each set of benchmarks will have:

- The raw `.json` results from pyperformance.
- Comparisons against important reference releases, as well as the merge base of the branch if `benchmark_base` was selected.  These include
  - A markdown table produced by `pyperf compare_to`.
  - A set of "violin" plots showing the distribution of results for each benchmark.

The most convenient way to get results locally is to clone this repo and `git pull` from it.

## Running benchmarks from the GitHub CLI

To automate benchmarking runs, it may be more convenient to use the [GitHub CLI](https://cli.github.com/).
Once you have `gh` installed and configured, you can run benchmarks by cloning this repository and then from inside it:

```bash
$ gh workflow run benchmark.yml -f fork=me -f ref=my_branch
```

Any of the parameters described above are available at the commandline using the `-f key=value` syntax.

## Costs

We are limited to 2,000 compute minutes per month.


| Action | Minutes |
| -- | -- |
| Benchmarks | 7 minutes (most of the time is on self-hosted runners) |
| CI | 10 minutes |

To reduce CI usage, PRs that are only documentation changes should add the `[skip ci]` token to their commit message.

# Results

The following is only a summary of certain key revisions. There is also a [complete list of results](results/README.md).

<!-- START table -->
## linux x86_64
| date | fork | ref | version | hash | vs. 3.10.4: | vs. 3.11.0: | vs. base: |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| [2023-02-07](results/bm-20230207-3.12.0a4+-dec1ab0) | python | main | 3.12.0a4+ | dec1ab0 | [1.29x faster \*](results/bm-20230207-3.12.0a4+-dec1ab0/bm-20230207-linux-x86_64-python-main-3.12.0a4+-dec1ab0-vs-3.10.4.md) | [1.03x faster \*](results/bm-20230207-3.12.0a4+-dec1ab0/bm-20230207-linux-x86_64-python-main-3.12.0a4+-dec1ab0-vs-3.11.0.md) |  |
| [2023-01-08](results/bm-20230108-3.12.0a3+-e47b139) | python | e47b13934b | 3.12.0a3+ | e47b139 | [1.30x faster \*](results/bm-20230108-3.12.0a3+-e47b139/bm-20230108-linux-x86_64-python-e47b13934b2eb50914e4-3.12.0a3+-e47b139-vs-3.10.4.md) | [1.03x faster \*](results/bm-20230108-3.12.0a3+-e47b139/bm-20230108-linux-x86_64-python-e47b13934b2eb50914e4-3.12.0a3+-e47b139-vs-3.11.0.md) |  |
| [2022-12-05](results/bm-20221205-3.12.0a2+-e3a3863) | python | e3a3863cb9 | 3.12.0a2+ | e3a3863 | [1.30x faster](results/bm-20221205-3.12.0a2+-e3a3863/bm-20221205-linux-x86_64-python-e3a3863cb9561705d3dd-3.12.0a2+-e3a3863-vs-3.10.4.md) | [1.03x faster](results/bm-20221205-3.12.0a2+-e3a3863/bm-20221205-linux-x86_64-python-e3a3863cb9561705d3dd-3.12.0a2+-e3a3863-vs-3.11.0.md) |  |
| [2022-10-22](results/bm-20221022-3.12.0a1+-f58631b) | python | main | 3.12.0a1+ | f58631b | [1.30x faster \*](results/bm-20221022-3.12.0a1+-f58631b/bm-20221022-linux-x86_64-python-main-3.12.0a1+-f58631b-vs-3.10.4.md) | [1.03x faster \*](results/bm-20221022-3.12.0a1+-f58631b/bm-20221022-linux-x86_64-python-main-3.12.0a1+-f58631b-vs-3.11.0.md) |  |
| [2022-10-06](results/bm-20221006-3.12.0a0-f8edc6f) | python | f8edc6ff53 | 3.12.0a0 | f8edc6f | [1.30x faster \*](results/bm-20221006-3.12.0a0-f8edc6f/bm-20221006-linux-x86_64-python-f8edc6ff531bb9885818-3.12.0a0-f8edc6f-vs-3.10.4.md) | [1.03x faster \*](results/bm-20221006-3.12.0a0-f8edc6f/bm-20221006-linux-x86_64-python-f8edc6ff531bb9885818-3.12.0a0-f8edc6f-vs-3.11.0.md) |  |
| [2022-12-06](results/bm-20221206-3.11.1-a7a450f) | python | a7a450f84a | 3.11.1 | a7a450f | [1.25x faster](results/bm-20221206-3.11.1-a7a450f/bm-20221206-linux-x86_64-python-a7a450f84a0874216031-3.11.1-a7a450f-vs-3.10.4.md) | [1.00x slower](results/bm-20221206-3.11.1-a7a450f/bm-20221206-linux-x86_64-python-a7a450f84a0874216031-3.11.1-a7a450f-vs-3.11.0.md) |  |
| [2022-10-24](results/bm-20221024-3.11.0-deaf509) | python | v3.11.0 | 3.11.0 | deaf509 | [1.26x faster](results/bm-20221024-3.11.0-deaf509/bm-20221024-linux-x86_64-python-v3.11.0-3.11.0-deaf509-vs-3.10.4.md) |  |  |
| [2022-09-11](results/bm-20220911-3.11.0rc2-ed7c3ff) | python | ed7c3ff156 | 3.11.0rc2 | ed7c3ff | [1.26x faster](results/bm-20220911-3.11.0rc2-ed7c3ff/bm-20220911-linux-x86_64-python-ed7c3ff15680c1939fad-3.11.0rc2-ed7c3ff-vs-3.10.4.md) | [1.00x slower](results/bm-20220911-3.11.0rc2-ed7c3ff/bm-20220911-linux-x86_64-python-ed7c3ff15680c1939fad-3.11.0rc2-ed7c3ff-vs-3.11.0.md) |  |
| [2022-08-05](results/bm-20220805-3.11.0rc1-41cb071) | python | 41cb07120b | 3.11.0rc1 | 41cb071 | [1.27x faster](results/bm-20220805-3.11.0rc1-41cb071/bm-20220805-linux-x86_64-python-41cb07120b7792eac641-3.11.0rc1-41cb071-vs-3.10.4.md) | [1.00x faster](results/bm-20220805-3.11.0rc1-41cb071/bm-20220805-linux-x86_64-python-41cb07120b7792eac641-3.11.0rc1-41cb071-vs-3.11.0.md) |  |
| [2022-07-25](results/bm-20220725-3.11.0b5-0771d71) | python | 0771d71eea | 3.11.0b5 | 0771d71 | [1.27x faster](results/bm-20220725-3.11.0b5-0771d71/bm-20220725-linux-x86_64-python-0771d71eea30316020a8-3.11.0b5-0771d71-vs-3.10.4.md) | [1.01x faster](results/bm-20220725-3.11.0b5-0771d71/bm-20220725-linux-x86_64-python-0771d71eea30316020a8-3.11.0b5-0771d71-vs-3.11.0.md) |  |
| [2022-07-11](results/bm-20220711-3.11.0b4-5a7e1e0) | python | 5a7e1e0a92 | 3.11.0b4 | 5a7e1e0 | [1.28x faster](results/bm-20220711-3.11.0b4-5a7e1e0/bm-20220711-linux-x86_64-python-5a7e1e0a92622c605ab2-3.11.0b4-5a7e1e0-vs-3.10.4.md) | [1.01x faster](results/bm-20220711-3.11.0b4-5a7e1e0/bm-20220711-linux-x86_64-python-5a7e1e0a92622c605ab2-3.11.0b4-5a7e1e0-vs-3.11.0.md) |  |
| [2022-06-01](results/bm-20220601-3.11.0b3-eb0004c) | python | main | 3.11.0b3 | eb0004c | [1.30x faster \*](results/bm-20220601-3.11.0b3-eb0004c/bm-20220601-linux-x86_64-python-main-3.11.0b3-eb0004c-vs-3.10.4.md) | [1.02x faster \*](results/bm-20220601-3.11.0b3-eb0004c/bm-20220601-linux-x86_64-python-main-3.11.0b3-eb0004c-vs-3.11.0.md) |  |
| [2022-05-30](results/bm-20220530-3.11.0b2-72f00f4) | python | main | 3.11.0b2 | 72f00f4 | [1.29x faster \*](results/bm-20220530-3.11.0b2-72f00f4/bm-20220530-linux-x86_64-python-main-3.11.0b2-72f00f4-vs-3.10.4.md) | [1.02x faster \*](results/bm-20220530-3.11.0b2-72f00f4/bm-20220530-linux-x86_64-python-main-3.11.0b2-72f00f4-vs-3.11.0.md) |  |
| [2022-05-06](results/bm-20220506-3.11.0b1-8d32a5c) | python | main | 3.11.0b1 | 8d32a5c | [1.29x faster \*](results/bm-20220506-3.11.0b1-8d32a5c/bm-20220506-linux-x86_64-python-main-3.11.0b1-8d32a5c-vs-3.10.4.md) | [1.01x faster \*](results/bm-20220506-3.11.0b1-8d32a5c/bm-20220506-linux-x86_64-python-main-3.11.0b1-8d32a5c-vs-3.11.0.md) |  |
| [2022-04-05](results/bm-20220405-3.11.0a7-2e49bd0) | python | main | 3.11.0a7 | 2e49bd0 | [1.24x faster \*](results/bm-20220405-3.11.0a7-2e49bd0/bm-20220405-linux-x86_64-python-main-3.11.0a7-2e49bd0-vs-3.10.4.md) | [1.02x slower \*](results/bm-20220405-3.11.0a7-2e49bd0/bm-20220405-linux-x86_64-python-main-3.11.0a7-2e49bd0-vs-3.11.0.md) |  |
| [2022-03-07](results/bm-20220307-3.11.0a6-3ddfa55) | python | main | 3.11.0a6 | 3ddfa55 | [1.20x faster \*](results/bm-20220307-3.11.0a6-3ddfa55/bm-20220307-linux-x86_64-python-main-3.11.0a6-3ddfa55-vs-3.10.4.md) | [1.06x slower \*](results/bm-20220307-3.11.0a6-3ddfa55/bm-20220307-linux-x86_64-python-main-3.11.0a6-3ddfa55-vs-3.11.0.md) |  |
| [2022-02-03](results/bm-20220203-3.11.0a5-c4e4b91) | python | main | 3.11.0a5 | c4e4b91 | [1.22x faster \*](results/bm-20220203-3.11.0a5-c4e4b91/bm-20220203-linux-x86_64-python-main-3.11.0a5-c4e4b91-vs-3.10.4.md) | [1.04x slower \*](results/bm-20220203-3.11.0a5-c4e4b91/bm-20220203-linux-x86_64-python-main-3.11.0a5-c4e4b91-vs-3.11.0.md) |  |
| [2022-01-13](results/bm-20220113-3.11.0a4-9471106) | python | main | 3.11.0a4 | 9471106 | [1.22x faster \*](results/bm-20220113-3.11.0a4-9471106/bm-20220113-linux-x86_64-python-main-3.11.0a4-9471106-vs-3.10.4.md) | [1.04x slower \*](results/bm-20220113-3.11.0a4-9471106/bm-20220113-linux-x86_64-python-main-3.11.0a4-9471106-vs-3.11.0.md) |  |
| [2021-12-08](results/bm-20211208-3.11.0a3-2e91dba) | python | main | 3.11.0a3 | 2e91dba | [1.20x faster \*](results/bm-20211208-3.11.0a3-2e91dba/bm-20211208-linux-x86_64-python-main-3.11.0a3-2e91dba-vs-3.10.4.md) | [1.06x slower \*](results/bm-20211208-3.11.0a3-2e91dba/bm-20211208-linux-x86_64-python-main-3.11.0a3-2e91dba-vs-3.11.0.md) |  |
| [2021-11-05](results/bm-20211105-3.11.0a2-e2b4e4b) | python | e2b4e4bab9 | 3.11.0a2 | e2b4e4b | [1.15x faster](results/bm-20211105-3.11.0a2-e2b4e4b/bm-20211105-linux-x86_64-python-e2b4e4bab90b69fbd361-3.11.0a2-e2b4e4b-vs-3.10.4.md) | [1.09x slower](results/bm-20211105-3.11.0a2-e2b4e4b/bm-20211105-linux-x86_64-python-e2b4e4bab90b69fbd361-3.11.0a2-e2b4e4b-vs-3.11.0.md) |  |
| [2021-10-05](results/bm-20211005-3.11.0a1-7c12e48) | python | 7c12e4835e | 3.11.0a1 | 7c12e48 | [1.12x faster](results/bm-20211005-3.11.0a1-7c12e48/bm-20211005-linux-x86_64-python-7c12e4835ebe52287acd-3.11.0a1-7c12e48-vs-3.10.4.md) | [1.12x slower](results/bm-20211005-3.11.0a1-7c12e48/bm-20211005-linux-x86_64-python-7c12e4835ebe52287acd-3.11.0a1-7c12e48-vs-3.11.0.md) |  |
| [2022-12-06](results/bm-20221206-3.10.9-1dd9be6) | python | 1dd9be6584 | 3.10.9 | 1dd9be6 | [1.00x slower](results/bm-20221206-3.10.9-1dd9be6/bm-20221206-linux-x86_64-python-1dd9be6584413fbfa823-3.10.9-1dd9be6-vs-3.10.4.md) | [1.26x slower](results/bm-20221206-3.10.9-1dd9be6/bm-20221206-linux-x86_64-python-1dd9be6584413fbfa823-3.10.9-1dd9be6-vs-3.11.0.md) |  |
| [2022-03-23](results/bm-20220323-3.10.4-9d38120) | python | v3.10.4 | 3.10.4 | 9d38120 |  | [1.26x slower](results/bm-20220323-3.10.4-9d38120/bm-20220323-linux-x86_64-python-v3.10.4-3.10.4-9d38120-vs-3.11.0.md) |  |
| [2021-05-03](results/bm-20210503-3.10.0a7+-d3b9134) | python | d3b9134ebb | 3.10.0a7+ | d3b9134 | [1.00x faster \*](results/bm-20210503-3.10.0a7+-d3b9134/bm-20210503-linux-x86_64-python-d3b9134ebb40bdb01ff5-3.10.0a7+-d3b9134-vs-3.10.4.md) | [1.25x slower \*](results/bm-20210503-3.10.0a7+-d3b9134/bm-20210503-linux-x86_64-python-d3b9134ebb40bdb01ff5-3.10.0a7+-d3b9134-vs-3.11.0.md) |  |

## darwin arm64
| date | fork | ref | version | hash | vs. 3.10.4: | vs. 3.11.0: | vs. base: |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| [2023-02-04](results/bm-20230204-3.12.0a4+-5a2b984) | python | main | 3.12.0a4+ | 5a2b984 | [1.23x faster \*](results/bm-20230204-3.12.0a4+-5a2b984/bm-20230204-darwin-arm64-python-main-3.12.0a4+-5a2b984-vs-3.10.4.md) | [1.01x faster \*](results/bm-20230204-3.12.0a4+-5a2b984/bm-20230204-darwin-arm64-python-main-3.12.0a4+-5a2b984-vs-3.11.0.md) |  |
| [2023-01-08](results/bm-20230108-3.12.0a3+-e47b139) | python | e47b13934b | 3.12.0a3+ | e47b139 | [1.22x faster \*](results/bm-20230108-3.12.0a3+-e47b139/bm-20230108-darwin-arm64-python-e47b13934b2eb50914e4-3.12.0a3+-e47b139-vs-3.10.4.md) | [1.00x faster \*](results/bm-20230108-3.12.0a3+-e47b139/bm-20230108-darwin-arm64-python-e47b13934b2eb50914e4-3.12.0a3+-e47b139-vs-3.11.0.md) |  |
| [2022-12-05](results/bm-20221205-3.12.0a2+-e3a3863) | python | e3a3863cb9 | 3.12.0a2+ | e3a3863 | [1.18x faster](results/bm-20221205-3.12.0a2+-e3a3863/bm-20221205-darwin-arm64-python-e3a3863cb9561705d3dd-3.12.0a2+-e3a3863-vs-3.10.4.md) | [1.04x slower](results/bm-20221205-3.12.0a2+-e3a3863/bm-20221205-darwin-arm64-python-e3a3863cb9561705d3dd-3.12.0a2+-e3a3863-vs-3.11.0.md) |  |
| [2022-11-13](results/bm-20221113-3.12.0a1+-367f552) | python | 367f552129 | 3.12.0a1+ | 367f552 | [1.19x faster](results/bm-20221113-3.12.0a1+-367f552/bm-20221113-darwin-arm64-python-367f552129341796d75f-3.12.0a1+-367f552-vs-3.10.4.md) | [1.03x slower](results/bm-20221113-3.12.0a1+-367f552/bm-20221113-darwin-arm64-python-367f552129341796d75f-3.12.0a1+-367f552-vs-3.11.0.md) |  |
| [2022-10-24](results/bm-20221024-3.12.0a0-dfb5d27) | python | dfb5d272e6 | 3.12.0a0 | dfb5d27 | [1.20x faster](results/bm-20221024-3.12.0a0-dfb5d27/bm-20221024-darwin-arm64-python-dfb5d272e6b99c2c70c6-3.12.0a0-dfb5d27-vs-3.10.4.md) | [1.02x slower](results/bm-20221024-3.12.0a0-dfb5d27/bm-20221024-darwin-arm64-python-dfb5d272e6b99c2c70c6-3.12.0a0-dfb5d27-vs-3.11.0.md) |  |
| [2022-12-06](results/bm-20221206-3.11.1-a7a450f) | python | a7a450f84a | 3.11.1 | a7a450f | [1.22x faster](results/bm-20221206-3.11.1-a7a450f/bm-20221206-darwin-arm64-python-a7a450f84a0874216031-3.11.1-a7a450f-vs-3.10.4.md) | [1.00x faster](results/bm-20221206-3.11.1-a7a450f/bm-20221206-darwin-arm64-python-a7a450f84a0874216031-3.11.1-a7a450f-vs-3.11.0.md) |  |
| [2022-10-24](results/bm-20221024-3.11.0-deaf509) | python | deaf509e8f | 3.11.0 | deaf509 | [1.22x faster](results/bm-20221024-3.11.0-deaf509/bm-20221024-darwin-arm64-python-deaf509e8fc6e0363bd6-3.11.0-deaf509-vs-3.10.4.md) |  |  |
| [2022-09-11](results/bm-20220911-3.11.0rc2-ed7c3ff) | python | ed7c3ff156 | 3.11.0rc2 | ed7c3ff | [1.22x faster](results/bm-20220911-3.11.0rc2-ed7c3ff/bm-20220911-darwin-arm64-python-ed7c3ff15680c1939fad-3.11.0rc2-ed7c3ff-vs-3.10.4.md) | [1.00x faster](results/bm-20220911-3.11.0rc2-ed7c3ff/bm-20220911-darwin-arm64-python-ed7c3ff15680c1939fad-3.11.0rc2-ed7c3ff-vs-3.11.0.md) |  |
| [2022-08-05](results/bm-20220805-3.11.0rc1-41cb071) | python | 41cb07120b | 3.11.0rc1 | 41cb071 | [1.23x faster](results/bm-20220805-3.11.0rc1-41cb071/bm-20220805-darwin-arm64-python-41cb07120b7792eac641-3.11.0rc1-41cb071-vs-3.10.4.md) | [1.01x faster](results/bm-20220805-3.11.0rc1-41cb071/bm-20220805-darwin-arm64-python-41cb07120b7792eac641-3.11.0rc1-41cb071-vs-3.11.0.md) |  |
| [2022-07-25](results/bm-20220725-3.11.0b5-0771d71) | python | 0771d71eea | 3.11.0b5 | 0771d71 | [1.23x faster](results/bm-20220725-3.11.0b5-0771d71/bm-20220725-darwin-arm64-python-0771d71eea30316020a8-3.11.0b5-0771d71-vs-3.10.4.md) | [1.01x faster](results/bm-20220725-3.11.0b5-0771d71/bm-20220725-darwin-arm64-python-0771d71eea30316020a8-3.11.0b5-0771d71-vs-3.11.0.md) |  |
| [2022-07-11](results/bm-20220711-3.11.0b4-5a7e1e0) | python | 5a7e1e0a92 | 3.11.0b4 | 5a7e1e0 | [1.23x faster](results/bm-20220711-3.11.0b4-5a7e1e0/bm-20220711-darwin-arm64-python-5a7e1e0a92622c605ab2-3.11.0b4-5a7e1e0-vs-3.10.4.md) | [1.01x faster](results/bm-20220711-3.11.0b4-5a7e1e0/bm-20220711-darwin-arm64-python-5a7e1e0a92622c605ab2-3.11.0b4-5a7e1e0-vs-3.11.0.md) |  |
| [2022-06-01](results/bm-20220601-3.11.0b3-eb0004c) | python | eb0004c271 | 3.11.0b3 | eb0004c | [1.21x faster](results/bm-20220601-3.11.0b3-eb0004c/bm-20220601-darwin-arm64-python-eb0004c27163ec089201-3.11.0b3-eb0004c-vs-3.10.4.md) | [1.01x slower](results/bm-20220601-3.11.0b3-eb0004c/bm-20220601-darwin-arm64-python-eb0004c27163ec089201-3.11.0b3-eb0004c-vs-3.11.0.md) |  |
| [2022-05-30](results/bm-20220530-3.11.0b2-72f00f4) | python | 72f00f420a | 3.11.0b2 | 72f00f4 | [1.21x faster](results/bm-20220530-3.11.0b2-72f00f4/bm-20220530-darwin-arm64-python-72f00f420afaba3bc873-3.11.0b2-72f00f4-vs-3.10.4.md) | [1.01x slower](results/bm-20220530-3.11.0b2-72f00f4/bm-20220530-darwin-arm64-python-72f00f420afaba3bc873-3.11.0b2-72f00f4-vs-3.11.0.md) |  |
| [2022-05-06](results/bm-20220506-3.11.0b1-8d32a5c) | python | 8d32a5c8c4 | 3.11.0b1 | 8d32a5c | [1.21x faster](results/bm-20220506-3.11.0b1-8d32a5c/bm-20220506-darwin-arm64-python-8d32a5c8c4e9c90b0a21-3.11.0b1-8d32a5c-vs-3.10.4.md) | [1.01x slower](results/bm-20220506-3.11.0b1-8d32a5c/bm-20220506-darwin-arm64-python-8d32a5c8c4e9c90b0a21-3.11.0b1-8d32a5c-vs-3.11.0.md) |  |
| [2022-04-05](results/bm-20220405-3.11.0a7-2e49bd0) | python | 2e49bd06c5 | 3.11.0a7 | 2e49bd0 | [1.22x faster](results/bm-20220405-3.11.0a7-2e49bd0/bm-20220405-darwin-arm64-python-2e49bd06c5ffab7d1540-3.11.0a7-2e49bd0-vs-3.10.4.md) | [1.01x slower](results/bm-20220405-3.11.0a7-2e49bd0/bm-20220405-darwin-arm64-python-2e49bd06c5ffab7d1540-3.11.0a7-2e49bd0-vs-3.11.0.md) |  |
| [2022-03-07](results/bm-20220307-3.11.0a6-3ddfa55) | python | 3ddfa55df4 | 3.11.0a6 | 3ddfa55 | [1.17x faster](results/bm-20220307-3.11.0a6-3ddfa55/bm-20220307-darwin-arm64-python-3ddfa55df48a67a5972f-3.11.0a6-3ddfa55-vs-3.10.4.md) | [1.04x slower](results/bm-20220307-3.11.0a6-3ddfa55/bm-20220307-darwin-arm64-python-3ddfa55df48a67a5972f-3.11.0a6-3ddfa55-vs-3.11.0.md) |  |
| [2022-02-03](results/bm-20220203-3.11.0a5-c4e4b91) | python | c4e4b91557 | 3.11.0a5 | c4e4b91 | [1.09x faster](results/bm-20220203-3.11.0a5-c4e4b91/bm-20220203-darwin-arm64-python-c4e4b91557f18f881f39-3.11.0a5-c4e4b91-vs-3.10.4.md) | [1.12x slower](results/bm-20220203-3.11.0a5-c4e4b91/bm-20220203-darwin-arm64-python-c4e4b91557f18f881f39-3.11.0a5-c4e4b91-vs-3.11.0.md) |  |
| [2022-01-13](results/bm-20220113-3.11.0a4-9471106) | python | 9471106fd5 | 3.11.0a4 | 9471106 | [1.17x faster](results/bm-20220113-3.11.0a4-9471106/bm-20220113-darwin-arm64-python-9471106fd5b47418ffd2-3.11.0a4-9471106-vs-3.10.4.md) | [1.04x slower](results/bm-20220113-3.11.0a4-9471106/bm-20220113-darwin-arm64-python-9471106fd5b47418ffd2-3.11.0a4-9471106-vs-3.11.0.md) |  |
| [2021-12-08](results/bm-20211208-3.11.0a3-2e91dba) | python | 2e91dba437 | 3.11.0a3 | 2e91dba | [1.15x faster](results/bm-20211208-3.11.0a3-2e91dba/bm-20211208-darwin-arm64-python-2e91dba437fe5c56c6f8-3.11.0a3-2e91dba-vs-3.10.4.md) | [1.06x slower](results/bm-20211208-3.11.0a3-2e91dba/bm-20211208-darwin-arm64-python-2e91dba437fe5c56c6f8-3.11.0a3-2e91dba-vs-3.11.0.md) |  |
| [2021-11-05](results/bm-20211105-3.11.0a2-e2b4e4b) | python | e2b4e4bab9 | 3.11.0a2 | e2b4e4b | [1.15x faster](results/bm-20211105-3.11.0a2-e2b4e4b/bm-20211105-darwin-arm64-python-e2b4e4bab90b69fbd361-3.11.0a2-e2b4e4b-vs-3.10.4.md) | [1.05x slower](results/bm-20211105-3.11.0a2-e2b4e4b/bm-20211105-darwin-arm64-python-e2b4e4bab90b69fbd361-3.11.0a2-e2b4e4b-vs-3.11.0.md) |  |
| [2022-12-06](results/bm-20221206-3.10.9-1dd9be6) | python | 1dd9be6584 | 3.10.9 | 1dd9be6 | [1.00x slower](results/bm-20221206-3.10.9-1dd9be6/bm-20221206-darwin-arm64-python-1dd9be6584413fbfa823-3.10.9-1dd9be6-vs-3.10.4.md) | [1.22x slower](results/bm-20221206-3.10.9-1dd9be6/bm-20221206-darwin-arm64-python-1dd9be6584413fbfa823-3.10.9-1dd9be6-vs-3.11.0.md) |  |
| [2022-03-23](results/bm-20220323-3.10.4-9d38120) | python | v3.10.4 | 3.10.4 | 9d38120 |  | [1.22x slower](results/bm-20220323-3.10.4-9d38120/bm-20220323-darwin-arm64-python-v3.10.4-3.10.4-9d38120-vs-3.11.0.md) |  |


<!-- END table -->

`*` indicates that the exact same versions of pyperformance was not used.
