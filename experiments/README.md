# Main experiments

## Layout

```
experiments/
├── builds/              # build-time configs and runner
│   ├── configs/         # one *.build.yaml per dataset alias
│   └── scripts/
│       ├── run_builds.sh
│       ├── filter_config.py
│       └── rewrite_results_dir.py
├── latency/             # single-thread per-query timing (compute_stats_latency)
│   ├── configs/         # canonical per-dataset *.search.yaml
│   ├── scripts/run_latency.sh
│   └── results/<dataset>/<method>/<build>/<variant>/<search>/latency_*.csv
├── multi_latency/       # multi-thread per-query timing (compute_stats_multi_latency)
│   ├── scripts/run_multi_latency.sh
│   └── results/...
└── batch/               # batch search_all throughput (compute_stats_batch)
    ├── scripts/run_batch.sh
    └── results/...
```

The `latency/configs/` YAMLs contain the search-related params for the
search sweep. `run_multi_latency.sh` and `run_batch.sh`
rewrite their `results_dir` on the fly (latency → multi_latency / batch),
so you only edit one set of files when adding a new sweep point.

TODO: get rid of the above, and finalize the configs

## Where do indices live?

The build step writes each index binary to:

```
${REPO_ROOT}/results/indexes/<dataset>/<method>/<build_name>/index.bin
```

Only the (large) binaries live there. The plotting CSVs and figures stay
under `experiments/<stage>/results/<dataset>/...`. 


## Datasets

Each runner accepts `--dataset` taking either:

* a top-level alias — `beir5`, `nq`, `hotpotqa`, `nq500k`, `quora`,
  `vidore` — which loads the corresponding YAML, or
* a single BEIR5 dataset name — `nfcorpus`, `scifact`, `arguana`,
  `scidocs`, `fiqa` — which filters `beir5.*.yaml` to that one dataset
  via `experiments/builds/scripts/filter_config.py`.

`--method` takes one of `mvivf`, `muvera`, `vamana`, `svh_graph`,
`fastplaid` (or `all`, the default).

## Method settings

Each method has its own search pipeline. Below, **columns are stages** (left
to right); each cell lists the YAML knobs / `variants[]` entries that apply
at that stage. Structural builds use a single `index.bin` per
`(method, build_name)`; search-time quantization is selected via
`variants[].quantizer` and sweep axes live under `search_configs`.

### TODO: TQ8-bit centers at MVIVF build + reranking

Not wired yet in these configs or consistently in pybind:

* **Build:** optional EightBitTQ / quantized centroid representation when
  constructing the MVIVF skeleton (**pending**).
* **Search:** reranking stage using EightBitTQ-style full-precision or TQ8
  scoring budgets via `num_rerank` / rerank path (**pending**).

Until then, MVIVF builds omit `pq_method` / `quantize_centers`; rerank rows in
the YAMLs are placeholders for the eventual sweep.

---

### MVIVF

IVF-style navigation over a K-means tree, then per-leaf scoring; timers split
roughly into beam/graph vs leaf vs rerank (see `methods.yaml` → `mvivf`).

| Build (tree skeleton) | IVF / beam navigation | Leaf retrieval & approximate scoring | Alternate variant (1-bit leaf path) | Full-precision rerank |
|-----------------------|----------------------|-------------------------------------|---------------------------------------|------------------------|
| `build_params`: `k_per_level`, `max_leaf_size`, `max_depth`, `niters`, `s`, `max_point_clouds_per_cluster` — **no** `pq_method` / `quantize_centers` in build YAML today | Sweep `nprobes` (and related beam knobs from `SearchParams`) | Variant **`tq4_search`**: `quantizer: TQ` (4-bit TurboQuant on leaf distances) | Variant **`tq1_probing`**: `quantizer: OneBitTQ` | **`num_rerank`** in `search_configs[].config[]`; **TODO** hook TQ8-bit rerank once build + bindings land |

---

### MUVERA

Learned FDE embeddings, graph index over documents, query-time quantization,
then rerank.

| Build (graph + FDE training layout) | Query embedding / quantization | Graph greedy search | Shortlist rerank |
|-------------------------------------|------------------------------|---------------------|------------------|
| `build_params`: `d_fde`, `R`, `L` (Vamana graph construction over FDE space) | Variant uses `quantizer: TQ` for the distance model on queries/candidates | Sweep main axis **`L`** (`methods.yaml` → `variable_param`) | **`num_rerank`** via `config[]`; map to MUVERA rerank budget in `SearchParams` |

MUVERA does **not** share MVIVF’s “leaf probing” vs “beam on centroids” split;
the plot breakdown follows `methods.yaml` → `muvera` labels (`t_fde`,
`t_quant`, `t_search`, `t_rerank`, …).

---

### MV-Vamana

Single graph over raw or PQ-compressed vectors; greedy search then rerank.

| Build (graph topology) | Query quantization | Greedy graph search | Rerank |
|--------------------------|-------------------|---------------------|--------|
| `build_params`: **`R`** (and family defaults) | Variant **`quantizer: TQ`** (4-bit) for approximate distances | Sweep **`L`** | **`num_rerank`** in `config[]` |

---

### SVH Graph

Single-vector **graph** index (Vamana-style construction on SVH points), then
greedy graph search, neighbor aggregation, rerank. This is `IndexSVHGraph*`, not
the IVF tree (`IndexSVHIVF` / `svh_ivf`).

| Build (graph `ann` config) | Query quantization | Greedy graph search | Aggregate & rerank |
|-----------------------------|--------------------|--------------------|--------------------|
| `build_params`: **`R`**, **`L`** (and defaults: `alpha`, `num_pass`, …) — see `IndexParams::svh_graph` | Variant **`quantizer: TQ`** (4-bit) for approximate distances | Sweep search-beam **`L`** (`methods.yaml` → `variable_param` is `L`) | **`num_rerank`**, optional `cut`; timers: `t_graph_search`, `t_aggregate`, `t_rerank` |

---

### FastPlaid (BEIR5 only)

External IVF + PQ product quantization; batch API differs from native MVSIC.

| Build (PQ codebook + IVF layout) | Coarse IVF (`n_ivf_probe`) | PQ distance / candidate generation | Full-score rerank |
|-----------------------------------|---------------------------|-----------------------------------|-------------------|
| `build_params`: **`nbits`** (e.g. 4); artifacts under `results/indexes/.../fastplaid/<build>/` | Sweep **`n_ivf_probe`** | FastPlaid internal PQ scoring | **`num_rerank`** → mapped to **`n_full_scores`** in `FastPlaidWrapper` |

---

### Quantizer name mapping

Shared vocabulary (`benchmarks/methods.yaml`, `benchmark_search.py`):

* `TQ` / `TQ4bit` → `TurboQuant` (4-bit)
* `OneBitTQ` / `TQ1bit` → `OneBitTQ`
* `EightBitTQ` / `TQ8bit` → `EightBitTQ`
* `PQ`, `FastScan`, `RaBitQ`, `SPQTQ` where the bound index family exposes them

## Status

Per-method × per-dataset run tracker. Tick a box once
`experiments/<stage>/results/<dataset>/<method>/<build>/<variant>/...`
contains the CSV.

### Builds

| Dataset    | mvivf | muvera | vamana | svh_graph | fastplaid |
|------------|:-:|:-:|:-:|:-:|:-:|
| nfcorpus   | [ ] | [ ] | [ ] | [ ] | [ ] |
| scifact    | [ ] | [ ] | [ ] | [ ] | [ ] |
| arguana    | [ ] | [ ] | [ ] | [ ] | [ ] |
| scidocs    | [ ] | [ ] | [ ] | [ ] | [ ] |
| fiqa       | [ ] | [ ] | [ ] | [ ] | [ ] |
| nq         | [ ] | [ ] | [ ] | [ ] |  —  |
| hotpotqa   | [ ] | [ ] | [ ] | [ ] |  —  |
| nq500k     | [ ] | [ ] | [ ] | [ ] |  —  |
| quora      | [ ] | [ ] | [ ] | [ ] |  —  |
| vidore     | [ ] | [ ] | [ ] | [ ] |  —  |

### Latency (single-thread)

| Dataset    | mvivf | muvera | vamana | svh_graph | fastplaid |
|------------|:-:|:-:|:-:|:-:|:-:|
| nfcorpus   | [ ] | [ ] | [ ] | [ ] | [ ] |
| scifact    | [ ] | [ ] | [ ] | [ ] | [ ] |
| arguana    | [ ] | [ ] | [ ] | [ ] | [ ] |
| scidocs    | [ ] | [ ] | [ ] | [ ] | [ ] |
| fiqa       | [ ] | [ ] | [ ] | [ ] | [ ] |
| nq         | [ ] | [ ] | [ ] | [ ] |  —  |
| hotpotqa   | [ ] | [ ] | [ ] | [ ] |  —  |
| nq500k     | [ ] | [ ] | [ ] | [ ] |  —  |
| quora      | [ ] | [ ] | [ ] | [ ] |  —  |
| vidore     | [ ] | [ ] | [ ] | [ ] |  —  |

### Multi-latency (per-query, ambient parlay pool)

FastPlaid does not support this mode.

| Dataset    | mvivf | muvera | vamana | svh_graph |
|------------|:-:|:-:|:-:|:-:|
| nfcorpus   | [ ] | [ ] | [ ] | [ ] |
| scifact    | [ ] | [ ] | [ ] | [ ] |
| arguana    | [ ] | [ ] | [ ] | [ ] |
| scidocs    | [ ] | [ ] | [ ] | [ ] |
| fiqa       | [ ] | [ ] | [ ] | [ ] |
| nq         | [ ] | [ ] | [ ] | [ ] |
| hotpotqa   | [ ] | [ ] | [ ] | [ ] |
| nq500k     | [ ] | [ ] | [ ] | [ ] |
| quora      | [ ] | [ ] | [ ] | [ ] |
| vidore     | [ ] | [ ] | [ ] | [ ] |

### Batch (search_all throughput)

| Dataset    | mvivf | muvera | vamana | svh_graph | fastplaid |
|------------|:-:|:-:|:-:|:-:|:-:|
| nfcorpus   | [ ] | [ ] | [ ] | [ ] | [ ] |
| scifact    | [ ] | [ ] | [ ] | [ ] | [ ] |
| arguana    | [ ] | [ ] | [ ] | [ ] | [ ] |
| scidocs    | [ ] | [ ] | [ ] | [ ] | [ ] |
| fiqa       | [ ] | [ ] | [ ] | [ ] | [ ] |
| nq         | [ ] | [ ] | [ ] | [ ] |  —  |
| hotpotqa   | [ ] | [ ] | [ ] | [ ] |  —  |
| nq500k     | [ ] | [ ] | [ ] | [ ] |  —  |
| quora      | [ ] | [ ] | [ ] | [ ] |  —  |
| vidore     | [ ] | [ ] | [ ] | [ ] |  —  |

## Running

All commands are run from `${REPO_ROOT}`. Indices are reused across
stages.

```bash
# 1) Build indices (writes to results/indexes/<dataset>/<method>/<build>/).
experiments/builds/scripts/run_builds.sh --dataset beir5
experiments/builds/scripts/run_builds.sh --dataset nq --method mvivf
experiments/builds/scripts/run_builds.sh --dataset arguana --method muvera

# 2) Single-thread latency.
experiments/latency/scripts/run_latency.sh --dataset beir5
experiments/latency/scripts/run_latency.sh --dataset nq --method mvivf

# 3) Multi-thread per-query latency (ambient parlay pool).
experiments/multi_latency/scripts/run_multi_latency.sh --dataset beir5

# 4) Batch throughput.
experiments/batch/scripts/run_batch.sh --dataset beir5
experiments/batch/scripts/run_batch.sh --dataset beir5 --method fastplaid
```

## Plots

Each search stage has a `scripts/plot.py` that reads its `results/`
tree and emits:

* a Pareto plot (recall@k vs. QPS, one curve per method × variant), and
* a per-method breakdown bar plot built from the timer labels in
  `benchmarks/methods.yaml` (the exact stages depend on the method — e.g.
  MVIVF splits beam / leaf / rerank; MUVERA uses FDE / quant / search /
  rerank).

Build-side reporting lives in
`experiments/builds/scripts/build_report.py`: a markdown table + bar
plots of build time and on-disk index size (MB) per dataset × method.
