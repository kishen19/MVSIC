# MVSIC: Multi-Vector Search, Indexing and Clustering

MVSIC (pronounced "music") is a high-performance C++ library with Python bindings for efficient multi-vector retrieval. It is designed for searching through datasets where each item is represented by a *set* of vectors (a "point cloud") rather than a single vector, typically generated from multi-vector embedding models like ColBert and ColPali. Similarity between point clouds is measured using the *chamfer similarity* measure (also known as the MaxSim operator).

MVSIC provides implementations of several state-of-the-art algorithms and also includes a comprehensive benchmarking suite to evaluate and compare their performance.

## Features

*   Built in C++ for speed and efficiency, leveraging the [ParlayLib](https://github.com/cmuparlay/parlaylib/) library for parallelism.
*   A user-friendly Python API for easy integration into existing workflows, powered by [pybind11](https://github.com/pybind/pybind11).
*   Implements several novel and baseline multi-vector ANN algorithms.
*   Supports various quantization methods (PQ + FastScan, RaBitQ) to reduce memory footprint and accelerate search.
*   Natively supports both **L2 (Euclidean)** and **Inner Product (MIPS)** distances for Chamfer distance calculations.
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

*   **Build:** [Bazel](https://bazel.build/) (version 6.x or later recommended) and a C++ compiler supporting C++17 (e.g., GCC 9+).
*   **Python:** Python 3.8+ and `pip`.
*   **Python Packages:** See `requirements.txt`.

### Installation

It is highly recommended to use a Python virtual environment.

```bash
# 1. Build the MVSIC C++ core and Python bindings using Bazel
# This creates a wheel file in the bazel-bin/mvsic/ directory
bazel build ...

# 2. Install the Python requirements and the MVSIC wheel
python3 -m pip install -r requirements.txt
```

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

Runnable Python examples for the core library live under `examples/`:

*   `examples/mvivf_example.py` — build and search an MVIVF index directly via the Python API.
*   `examples/muvera_example.py` — build and search a MUVERA index.
*   `examples/vamana_example.py` — build and search a Multi-Vector Vamana index.
*   `examples/fast_plaid_benchmark.py` — compare against the external FastPlaid baseline.

## Benchmarking

There are two ways to benchmark MVSIC's algorithms:

*   **`benchmarks/`** — lower-level scripts (`benchmark_build.py`, `benchmark_search.py`) driven by YAML configs under `benchmarks/configs/` and `benchmarks/plot_configs/`. These are the scripts the higher-level runners below wrap.
*   **`experiments/`** (recommended) — the full suite used to produce the paper's results: build indices, sweep search parameters (latency and batch/throughput modes), and generate Pareto (recall vs. QPS) plots, including comparisons against external baselines. From the repo root:

    ```bash
    experiments/builds/scripts/run_builds.sh --dataset beir5
    experiments/latency/scripts/run_latency.sh --dataset beir5
    experiments/batch/scripts/run_batch.sh --dataset beir5
    experiments/plot_all.sh
    ```

    See `experiments/README.md` for the full parameter cheatsheet, dataset aliases, and per-method configuration details.