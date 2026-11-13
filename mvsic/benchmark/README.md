# Installed MVIVF benchmarks

Install the package, create the existing build/search YAML files, and run them
from your working directory. No source checkout, git metadata, schema conversion,
or manually selected single-grid entry is needed.

```sh
pip install /path/to/MVSIC
mvsic benchmark validate --config mvivf.build.yaml
mvsic benchmark build --config mvivf.build.yaml --build-threads 8
mvsic benchmark search --config mvivf.search.yaml --mode latency
mvsic benchmark search --config mvivf.search.yaml --mode multi_latency --num_threads 8
mvsic benchmark search --config mvivf.search.yaml --mode batch --num_threads 8
pip install 'mvsic[plot]'
mvsic benchmark plot --config mvivf.plot.yaml
```

PyYAML is installed with MVSIC. CSV generation uses the standard library; plotting
requires the optional matplotlib extra. Copy the installed examples to edit:

```python
from importlib.resources import files
from pathlib import Path
for name in ("mvivf.build.yaml", "mvivf.search.yaml", "mvivf.plot.yaml"):
    Path(name).write_text(files("mvsic.benchmark").joinpath("configs", name).read_text())
```

The callable API uses the same workflow as the CLI:

```python
from mvsic.benchmark import load_config, run

build = load_config("mvivf.build.yaml")
print(build.validate())
run("build", build, build_threads=8)
search = load_config("mvivf.search.yaml")
search.run("search", mode="latency", query_subsample=1000, query_subsample_seed=42)
run("plot", load_config("mvivf.plot.yaml"))
```

`load_config(filename, *, base_dir=None)` returns a `BenchmarkPlan` with
`validate(action=None, **options)` and `run(action, **options)`. Module-level
`run(action, plan_or_filename, **options)` accepts the same options. The default
base is the caller's working directory, matching the original scripts, **not**
the directory containing the YAML. Use `--base-dir` or `base_dir=` to set it
explicitly. `validate` checks the complete plan without importing native code or
opening the corpus.

A structural build file uses the original schema:

```yaml
name: MVIVF build
datasets:
  - name: arguana
    path: data/beir/arguana
    index_dir: results/indices/arguana
    is_mmap: true
indices:
  - name: mvivf
    metric: ip
    build_configs:
      - name: tree
        params: {k_per_level: 0, max_leaf_size: 100, niters: 5, seed: 0}
```

The dataset contains `<name>_points.pcs`, `<name>_queries.pcs` and
`<name>_chamfer_neighbors.gt`. Build needs only the points file. Each named build
writes `<index_dir>/mvivf/<build_name>/{index.bin,index_params.json,build_stats.json}`.
The saved tree is structural: builds use float leaves and raw centers. Serving
representations are prepared when search loads that same tree. Build statistics
retain `build_time_sec`, `index_size_mb`, build parameters and tree height, with
additional process memory and provenance.

Search uses the existing separate `builds` schema:

```yaml
name: MVIVF search
datasets:
  - {name: arguana, path: data/beir/arguana, index_dir: results/indices/arguana,
     results_dir: results/search/arguana, is_mmap: true}
indices:
  - name: mvivf
    metric: ip
    builds:
      - build_name: tree
        build_params: {k_per_level: 0, max_leaf_size: 100, niters: 5, seed: 0}
        variants:
          - {name: tq1, compress: true, quantizer: OneBitTQ}
          - {name: tq4, compress: true, quantizer: TQ}
        search_configs:
          - name: k=10
            params:
              k: [10]
              nprobes: [1, 2, 4, 8, 16, 32]
              tq8_rerank: [true]
              query_compression: [Wards]
              query_compression_threshold: [0.5]
              config:
                - {name: nr20, num_rerank: 20}
                - {name: nr50, num_rerank: 50, nprobes: [4, 8, 16, 32, 64]}
```

All datasets, builds, serving variants and search configs are expanded. Each
named `params.config` row overrides the shared axes, then expands its own
Cartesian product; correlated choices are never combined with other named rows.
Points are sorted by `nprobes` as in the original runner. YAML anchors and explicit
overrides work; duplicate explicit keys fail. An omitted serving `variants` list
means raw centers/float leaves and an empty variant name. The earlier combined
`build_configs`/`params`/`search_configs` form is also accepted; `datasets[].results`
acts as the older shared index/result root when `index_dir` is absent.

Result paths preserve the original layout:

```
<results_dir>/mvivf/<build>/<variant>/<search>/latency_results_<config>.csv
<results_dir>/mvivf/<build>/<variant>/<search>/multi_latency_results_<config>.csv
<results_dir>/mvivf/<build>/<variant>/<search>/batch_results_<config>.csv
```

Empty variant/config names collapse their directory/suffix. `--latency` remains
an alias. Without a mode or thread override, the original dual measurement flow
produces `results_*.csv` with sequential-query and batch throughput. The older
`--num_threads N` mode produces `latency_results_*.csv` with within-query N-worker
latency. `--mode latency` always uses one worker; multi-latency and batch use
`--num_threads`, `PARLAY_NUM_THREADS`, or the available CPU count in that order.
Construction uses `--build-threads`, then `PARLAY_NUM_THREADS`, then available CPUs.

These counts select Parlay concurrency. OpenMP, MKL, OpenBLAS, NumExpr, VecLib
and Rayon limits default to1; builds and parallel measurements preserve explicit
environment settings. Single-query latency forces all limits to1. Process
metadata records provided and effective limits. Build fingerprints include
effective construction limits, so changing only search concurrency does not
invalidate a reusable index.

The original stopping policy is retained per named sweep: stop at recall 1.0,
on the original three-equal-recall plateau rule after more than three points, on
a recall drop, or below `--QPS_thresh` (default 10; nonpositive disables it).
`--query_subsample` defaults to 1000 and `--query_subsample_seed` to 42; a
nonpositive limit uses all queries. Selection uses sorted NumPy seeded sampling
without replacement and is saved exactly. Latency defaults to one repetition
and ten warmup queries; batch defaults to three repetitions and one warmup batch.
`--repeats` and `--warmup-queries` override these. The no-mode dual flow defaults
to three repetitions and 100 warmup queries, matching the original policy.

CSV columns include the original `k`, `recall_1_k`, `recall_k_k`, `QPS_seq`,
`QPS_par`, `nprobes`, `num_rerank` and stage labels, plus explicit timing scope.
Only measured QPS modes are populated; a single-thread measurement is not copied
into the parallel column. CSV and raw JSON retain full float precision. Native
search timing excludes Python/result conversion. Single-query stage diagnostics
are collected separately through `search_with_stats`; their timers do not add
Python overhead to native QPS. Batch has no invented per-query latency or stage
timings. All repetitions, predictions and query latencies are retained in NPZ
files under each result directory's `.mvsic/` subdirectory.

A raw record's `first_call_seconds` may include lazy TQ8 preparation. Its scope is
one query or an entire batch according to the mode. With zero warmup that call
is included in the first measured repetition. First-call and warmup totals
may overlap; do not add them. Search preparation and process RSS are recorded
separately. RSS includes corpus, Python/runtime, codecs and allocator storage,
and its high-water mark accumulates over the worker's sweep.

Build reuse and search resume verify content hashes, parameters, implementation
DSOs, environment and selected query rows. A serving variant can load the same
structural artifact without pretending it was built with that representation.
An original v4 artifact with `index_params.json` can also be loaded: declared
structural parameters and the artifact are checked, and its unverified original
build provenance is recorded explicitly. No new build timing is invented.
Changing build parameters or corpus requires a fresh build. A changed runtime or
search policy requires fresh measurements; `--force` replaces incompatible CSVs
and raw results, while `--rebuild` rebuilds indices. `append: false` also replaces
that search config's results. Failures retain logs and return a nonzero exit
status; `--fail-fast` stops after the first failure. Original `--methods` metadata
is optional because the supported method is installed with the framework.

Supported retrieval is MVIVF IP (negative mean MaxSim), with `None`, `TQ`,
`OneBitTQ`, and `EightBitTQ` leaves and raw/TQ4 centers. Retired PQ/FastScan/RaBitQ/
SPQTQ variants, other algorithm families and L2 retrieval fail explicitly instead
of being silently skipped. Legacy `pq_method: 0` and `4` are accepted for combined
float/TQ configs. Serving TQ1 retains its native norm checks.

Assignment quantization is independent: `assignment_quantizer` accepts `None`
(float), `TQ8`, `TQ4`, `TQ1`; centroid updates remain float. `build_with_8btq` is
an existing alias for TQ8 assignment. Both build and search `build_params` must
state the same training choices. `inner_init` (`default`, `uniform`, `kmeans++`,
`prefix_doubling`) and `inner_backend` (`legacy`, `adaptive`) control spherical
IP clustering. There is no metric selector in the compiled API. Original YAML
`metric: ip` is accepted. Older experimental `inner_metric: same` or `ip` fields
are discarded during validation; `l2` fails explicitly before native work.

Plotting accepts both original top-level `results` and named `plots` groups,
including group-level `k`, with existing `--affix`, `--style` and `--format`
options. It produces Pareto figures and recall-target analysis CSVs under
`results/<plot name>`. Missing measurements remain blank. The internal version-1
job schema remains available for developer experiments; it is not required for
the public YAML workflow.
