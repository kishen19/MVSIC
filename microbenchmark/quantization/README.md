## Chamfer / Quantization Microbenchmarks

Microbenchmarks for evaluating Chamfer distance kernels, quantization schemes, and MVIVF-like search patterns. Most support both **synthetic** and **file** modes.

Build:

```bash
bazel build //microbenchmark/quantization/...
```

---

## Multi‑vector benchmarks

### `bench_chamfer_overretrieve`

**Goal**: Measure *quality* of approximate multi‑vector Chamfer distances at fixed candidate budgets, i.e. **Recall@k vs candidate budget \(k'\)**.

**Key methods**:
- FastScan
- TurboQuant 4‑bit (TQ4)
- TurboQuant‑PQ 4‑bit (TQPQ, B = 1/2/4/8)
- Optional PQ / RaBitQ (via flags)

**File mode (BEIR)**:

```bash
./bazel-bin/microbenchmark/quantization/bench_chamfer_overretrieve \
  -dist_func IP \
  -i data/beir/arguana/arguana_points.pcs \
  -q data/beir/arguana/arguana_queries.pcs \
  -gt data/beir/arguana/arguana_chamfer_neighbors.gt \
  -k 10 \
  -pq      \  # enable PQ   (optional)
  -rabitq     # enable RaBitQ (optional)
```

**Synthetic mode** (no `-i/-q`):

```bash
./bazel-bin/microbenchmark/quantization/bench_chamfer_overretrieve \
  -dist_func IP \
  -N_db 20000 -N_q 200 -K_db 64 -D 128 \
  -k 10
```

**Output**:
- Prints DB/Q stats.
- Reports **Recall@k** for a grid of candidate budgets \(k' \in \{k, 2k, 5k, 10k, 25k, 50k\}\).
- One row per method; values are averaged over queries.

---

### `bench_chamfer_topk`

**Goal**: Leaf‑style MVIVF baseline for **top‑k Chamfer search**. Simulates a k‑means tree by partitioning the DB into uniform “leaves” and measuring:
- **Exact baseline** (unquantized Chamfer),
- FastScan / TQ4 / TQPQ (B = 4/8) multi‑vector quantizers.

**Modes**:
- **File mode** (BEIR‑style `.pcs` files): requires `-i <db.pcs> -q <queries.pcs>`.
- **Synthetic mode** (default if `-i/-q` absent).

**File mode example (IP)**:

```bash
./bazel-bin/microbenchmark/quantization/bench_chamfer_topk \
  -dist_func IP \
  -i data/beir/arguana/arguana_points.pcs \
  -q data/beir/arguana/arguana_queries.pcs \
  -leaf_size 500 \
  -reps 1
```

**Synthetic example**:

```bash
./bazel-bin/microbenchmark/quantization/bench_chamfer_topk \
  -dist_func IP \
  -N_db 20000 -N_q 200 -K_db 64 -D 128 \
  -leaf_size 500 \
  -reps 1
```

**Output**:
- Train/encode times for FastScan, TQ4, TQPQ.
- **Baseline times** table with columns:
  - `#leaves`, `Exact [s]`, `FS [s]`, `TQ [s]`, `TQPQ4 [s]`, `TQPQ8 [s]`.
- **Baseline speedup** table: `Exact / Method` ratios.

---

## Single‑vector benchmarks

### `bench_overretrieve`

**Goal**: Single‑vector analog of `bench_chamfer_overretrieve`. Measures approximate quality (Recall@k) vs candidate budget for scalar/vector ANN methods.

Typical usage (synthetic):

```bash
./bazel-bin/microbenchmark/quantization/bench_overretrieve \
  -dist_func IP
```

---

### `bench_pq_fastscan`  (**TODO: rename**)

**Goal**: Compare **exact vs PQ vs FastScan vs TQ/TQPQ** on synthetic single‑vector data, focusing on **per‑query latency and speedups**.

(**TODO**: rename)

Example run (synthetic, IP):

```bash
./bazel-bin/microbenchmark/quantization/bench_pq_fastscan \
  -dist_func IP
```

**Output**:
- Per‑method times and speedups vs exact.
