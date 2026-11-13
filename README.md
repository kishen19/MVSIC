# MVSIC: Multi-Vector Search, Indexing and Clustering

MVSIC (pronounced "music") is a high-performance C++ library with Python bindings for efficient multi-vector retrieval. It is designed for searching through datasets where each item is represented by a *set* of vectors (a "point cloud") rather than a single vector, typically generated from multi-vector embedding models like ColBert and ColPali. Similarity between point clouds is measured using the *chamfer similarity* measure (also known as the MaxSim operator).

The installed package exposes the compiled MVIVF API. Other algorithm families
remain available as optional legacy C++ targets while their APIs are migrated.

Sources are organized by responsibility:

```
mvsic/core/                         # data, index contract, shared CPU kernels
mvsic/core/kmeans/                  # shared single-vector training and seeding
mvsic/core/kernels/                 # CPU capabilities, dispatch, Intel kernels
mvsic/core/quantization/            # quantizers and their encoded layouts
mvsic/core/distance_measures/       # common one-to-many and many-to-many scoring
mvsic/algorithms/mvivf/             # compiled MVIVF topology and serving kernels
mvsic/algorithms/mvivf/mvclustering/ # assignment and float centroid updates
mvsic/algorithms/{emdivf,flat,mpool,muvera,svh,vamana}/
mvsic/algorithms/benchmark.cc       # common native timing loop over Index
mvsic/benchmark/                    # installed config-based benchmark interface
mvsic/python/                      # one native binding translation unit
```

MVClustering belongs to MVIVF because it trains point-cloud centers for that
method. Single-vector k-means remains in `core/kmeans/` because other index and
quantization implementations also use it. Small Intel dot kernels share one
source file; the larger serving and clustering kernels compile into isolated
AVX2, AVX-512 and VNNI libraries from the same private sources. Selection happens
when an operation is prepared, outside the scoring loops.

## Features

*   Built in C++ for speed and efficiency, leveraging the [ParlayLib](https://github.com/cmuparlay/parlaylib/) library for parallelism.
*   A user-friendly Python API for easy integration into existing workflows, powered by [pybind11](https://github.com/pybind/pybind11).
*   Implements several novel and baseline multi-vector ANN algorithms.
*   Compiled MVIVF supports float, TQ8, TQ4 and TQ1 leaves, with independent assignment quantization during training. Optional legacy C++ targets also retain PQ, FastScan and RaBitQ.
*   Uses **Inner Product / MaxSim** for retrieval and MVClustering, with spherical inner k-means updates.
*   Includes a benchmarking suite for comparing the QPS-Recall performance of different indexing methods.

## Algorithms

MVSIC implements the following multi-vector indexing algorithms natively:

*   **MVIVF:** A novel Multi-Vector Inverted File index that uses a k-means-like tree structure built on ``centroids'' point clouds. Also available in `Flat` and `Spill` variants.
*   **MUVERA:** Encodes point clouds into single vectors using Fixed-Dimensional Encodings (FDEs) and then uses a fast Vamana graph for retrieval.
*   **Multi-Vector Vamana:** A baseline implementation of the graph-based DiskANN algorithm that uses direct Chamfer distance for comparisons.
*   **MPool:** A baseline that performs mean-pooling over the point clouds to produce single vectors and indexes them with a Vamana graph.
*   **SVH (IVF and Graph):** Baselines that index single-vector heuristics (SVHIVF, SVHGraph) derived from the point clouds.

The `experiments/` benchmarking suite additionally compares against several **external baselines** (not implemented in this repo, but built/run as separate dependencies): **FastPlaid**, **IGP**, **hnswlib**, and **GEM**. See `experiments/README.md` for how these are configured and run.

## Setup & Installation

### Dependencies

* **Build:** CMake 3.24+, a C++20 compiler, OpenMP, and a Linux x86-64 host.
  Native dependencies are pinned by CMake; source-directory overrides support
  offline builds. Bazel remains available for native development.
* **CPU:** x86-64-v3 (AVX2/FMA) minimum. Separate AVX-512 and VNNI libraries are
  selected when both CPU capabilities and OS vector state allow them.
* **Python:** Python 3.10+, NumPy, and PyYAML. Plotting is an optional extra.

### Installation

It is highly recommended to use a Python virtual environment.

```bash
python -m pip install .
# Optional plotting:
python -m pip install '.[plot]'
# Editable development installation:
python -m pip install --no-build-isolation -e .
```

PEP 517 builds a CPython/platform-tagged wheel. It includes the native libraries,
Python module, benchmark commands, and example configuration. It does not install
external baselines or embedding models. Set `CMAKE_BUILD_PARALLEL_LEVEL` to limit
compiler concurrency on shared machines.

```python
import numpy as np
import mvsic

values = np.random.default_rng(0).normal(size=(100, 8, 128)).astype(np.float32)
values /= np.linalg.norm(values, axis=-1, keepdims=True)
points = mvsic.PointCloudSet(values)
params = mvsic.IndexParams.mvivf(max_leaf_size=20)
params.mvclus.assignment_quantizer = mvsic.ClusteringQuantizer.TQ8
index = mvsic.IndexMVIVF(128, params)
index.build(points)
search = mvsic.SearchParams.mvivf(k=10, nprobes=8, num_rerank=20)
ids, distances = index.search(values[0], search)
batch_ids, batch_distances = index.search_all(mvsic.PointCloudSet(values[:10]), search)
index.save("example.mvivf")
```

Retrieval scores are negative directed mean MaxSim; lower scores rank first.
`IndexMVIVF` and standalone MVClustering use IP without a metric selector.
The original YAML `metric: ip` remains supported; L2 requests fail explicitly.
Private k-means still supports L2 for Euclidean quantizer codebooks.
Serving supports float (`None_`), TQ4
(`TQ`), TQ1 (`OneBitTQ`), and TQ8 (`EightBitTQ`) leaves, with float or TQ4 centers.
TQ1 serving requires unit-norm corpus and original query vectors; the boundary
rejects other norms. Clustering's independent TQ1 assignment retains general
norms, including zero vectors, and every assignment mode updates centroids in
float with the existing document weights and spherical normalization.

`PointCloudSet` borrows contiguous float32 NumPy values by default and retains
their owner. Do not modify bound arrays during the index lifetime; `copy=True`
creates independent values. Explicit ragged offsets are contiguous uint64 and
count **float coordinates**, and optional IDs are contiguous uint32, excluding
the reserved missing-result ID `UINT32_MAX`. Returned
cloud views are read-only. C++ views without an owner require the caller to keep
their values, offsets, and IDs alive and unchanged. Batched result arrays have
shape `(queries, k)`; missing neighbors use ID `UINT32_MAX` and distance infinity.

The native v4 index file preserves compatibility with original MVIVF topology.
It contains float centers and document IDs, but no metric, dimension, or codec
model metadata. Load with the same constructor metric/dimension and corpus;
serving models and encoded data are deterministically prepared again. The TQ8
rerank cache is prepared once on its first request and is reset on build/load. The
benchmark artifact manifest checks structural configuration and input fingerprints;
the same saved tree can be loaded with different serving quantizers.

`MVSIC_CPU_ISA=avx2|avx512|vnni` selects a backend at preparation boundaries for
testing; unavailable requests raise an error. `mvsic.capabilities()` distinguishes
hardware, usable OS state, and compiled implementations. The override applies
to serving and clustering kernels; RaBitQ rotation uses its own hardware/OS
dispatch. AMX has a capability placeholder only. VNNI and AMX have not been
hardware-validated on this machine.

## I/O Format

MVSIC uses a specific binary format for storing point cloud datasets to enable efficient loading and memory mapping. The file is structured as follows:

| Type    | Name                | Size                                           | Description                                                                                             |
| :------ | :------------------ | :--------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| `size_t`  | `dimension`         | `sizeof(size_t)`                               | The dimension of each vector in the point clouds.                                                       |
| `size_t`  | `num_point_clouds`  | `sizeof(size_t)`                               | The total number of point clouds in the dataset.                                                        |
| `size_t`  | `num_total_vectors` | `sizeof(size_t)`                               | The total number of individual vectors across all point clouds.                                         |
| `float[]` | `vectors`           | `num_total_vectors * dimension * sizeof(float)`  | A flat array containing the concatenated vector data for all point clouds.                              |
| `size_t`  | `num_offsets`       | `sizeof(size_t)`                               | The number of offsets, which should be `num_point_clouds + 1`.                                          |
| `size_t[]`| `offsets`           | `num_offsets * sizeof(size_t)`                 | An array of offsets indicating the starting position of each point cloud's data in the `vectors` array. |

The data for the i-th point cloud (0-indexed) is located in the `vectors` array from index `offsets[i]` up to (but not including) `offsets[i+1]`. The number of vectors in the i-th point cloud is `(offsets[i+1] - offsets[i]) / dimension`.


## Usage Examples

Python examples live under `examples/`:

*   `examples/mvivf_example.py` — build and search an MVIVF index directly via the Python API.
*   `examples/muvera_example.py` — legacy MUVERA API; requires the original multi-family Python extension.
*   `examples/vamana_example.py` — legacy Multi-Vector Vamana API; requires the original multi-family Python extension.
*   `examples/fast_plaid_benchmark.py` — compare against the external FastPlaid baseline.

## Benchmarking

The installed package accepts the existing YAML benchmark workflow directly:

```bash
mvsic benchmark validate --config experiment.build.yaml
mvsic benchmark build --config experiment.build.yaml
mvsic benchmark search --config experiment.search.yaml --mode latency
mvsic benchmark search --config experiment.search.yaml --mode multi_latency --num_threads 8
mvsic benchmark search --config experiment.search.yaml --mode batch --num_threads 8
mvsic benchmark plot --config experiment.plot.yaml
```

The original `datasets`, named `build_configs`, `builds`, `variants`, and
`search_configs` remain the public format, with the same parameter grids and
output directories. YAML anchors work as before. Relative paths are resolved
from the working directory, with an explicit `--base-dir` override when needed.
No configuration conversion, Bazel invocation, or source checkout is needed to
run installed MVIVF benchmarks. Other methods require their own compiled adapters
before they can use this interface.

See [the benchmark interface](mvsic/benchmark/README.md) for installed YAML
examples, native latency versus batch measurement, fingerprints, resume, and
plotting. [Controlled comparisons](tests/refactor/README.md)
describe original/refactor measurements and compilation-cost scope.

The same workflow is callable from Python:

```python
from mvsic.benchmark import load_config, run

build = load_config("experiment.build.yaml")
run("build", build, build_threads=8)
run("search", "experiment.search.yaml", mode="latency")
run("plot", "experiment.plot.yaml")
```

All three measurement modes use `algorithms/benchmark.cc` through the public
`Index` contract. Single-thread latency executes with a one-worker Parlay
scheduler; parallel latency permits work within each query; batch throughput
uses the index's grouped traversal. This preserves the distinct workloads
without compiling a separate benchmark implementation for every method or
threading mode.

The historical experiment suite remains for legacy algorithms:

*   **`benchmarks/`** — lower-level scripts (`benchmark_build.py`, `benchmark_search.py`) driven by YAML configs under `benchmarks/configs/` and `benchmarks/plot_configs/`. These are the scripts the higher-level runners below wrap.
*   **`experiments/`** (recommended) — the full suite used to produce the paper's results: build indices, sweep search parameters (latency and batch/throughput modes), and generate Pareto (recall vs. QPS) plots, including comparisons against external baselines. From the repo root:

    ```bash
    experiments/builds/scripts/run_builds.sh --dataset beir5
    experiments/latency/scripts/run_latency.sh --dataset beir5
    experiments/batch/scripts/run_batch.sh --dataset beir5
    experiments/plot_all.sh
    ```

    See `experiments/README.md` for the full parameter cheatsheet, dataset aliases, and per-method configuration details.
