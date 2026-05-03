# MVIVF ablation runs

Default dataset in `run_ablation.sh` is `arguana`; override with `--dataset <name>` or `--datasets a,b` where needed.

Each stage uses YAML under `configs/` (`mvivf.stageN.{build,search}.yaml`). After we pick a winner for stage *N*, freeze those parameter values into the next stage's configs before continuing.

Stages 5/6/7 mix the **mvivf** and **mvivf_spill** index families inside the
same stage. Their results land directly under `results/<dataset>/stage<N>/`
(with one `<index_name>/` subtree per family) and the plotter splits each
curve by `(index_name, group_col)` so e.g. "mvivf / num_spill=1" is shown
next to "mvivf_spill / num_spill=4".

---

## Stages 1-4

Single-family `mvivf` sweeps; results land under `results/<dataset>/stage<N>/mvivf/`.

### Stage 1 - `k_per_level`

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 1 --dataset arguana --task all
```

### Stage 2 - `max_leaf_size`

(Update configs with the stage-1 `k_per_level` winner.)

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 2 --dataset arguana --task all
```

### Stage 3 - `max_depth`

(Update configs with stage-1 `k_per_level` and stage-2 `max_leaf_size` winners.)

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 3 --dataset arguana --task all
```

### Stage 4 - `niters`

(Update configs with winners from stages 1-3.)

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 4 --dataset arguana --task all
```

---

## Stages 5-7 (mixed mvivf + mvivf_spill)

Same `--variant mvivf` runner; the YAMLs declare both `mvivf` and
`mvivf_spill` indices so a single invocation builds/searches/evaluates both.
The plotter labels each curve by `(index_name, <group_col>)`.

### Stage 5 - `num_spill` (spill ratio)

Compares spilling strategies (`num_spill in {2, 4, 10}`, `num_spill_l2 = 1`)
against the base `mvivf` build (treated as `num_spill = 1` in the plot).

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 5 --dataset arguana --task all
```

### Stage 6 - `query_compression`

Toggles `query_compression` between `None` and `Wards` (ball-carving + Ward's
hierarchical query clustering, threshold `0.7`) for the stage-5 winner build.
Two curves per index family.

(Update configs with the stage-5 `num_spill` winner.)

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 6 --dataset arguana --task all
```

### Stage 7 - quantization (1BTQ vs FastScan)

Compares 1-bit TurboQuant (search-time variant on the raw skeleton) vs
FastScan PQ (requires a separate PQ-built skeleton with `pq_method: 3`,
`block_size: 8`, `num_clusters_per_block: 16`). Two builds per index family;
each one carries a single `variants:` block (`tq1bit` / `fastscan`).

(Update configs with the stage-5 `num_spill` winner and freeze the stage-6
`query_compression` choice.)

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 7 --dataset arguana --task all
```

---

## MVIVF Flat (optional)

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf_flat --stage 1 --dataset arguana --task all
```

---

## Tasks and options

| `--task`   | Meaning |
|-----------|---------|
| `all`     | build, then search, then evaluate (default) |
| `build`   | builds only |
| `search`  | latency search only |
| `evaluate`| summaries / plots from existing CSVs |

Examples:

```bash
# Build only for MVIVF stage 3
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 3 --dataset arguana --task build

# Evaluate only (after results exist)
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 5 --dataset fiqa --task evaluate
```

Optional: append `--with-winners` to pick budget winners during evaluate.

Indices default under `results/mvivf_ablation/indices/` per dataset entries in the YAMLs; search CSVs land under `experiments/mvivf_ablation/results/<dataset>/stage<N>/...`.

### Plotting tweaks for stages 5/6/7

`scripts/plot.py` accepts `--label-by <col>` (e.g. `index_name`) which
prefixes each curve label with the value of that column, so a single axis can
mix `mvivf` and `mvivf_spill` rows. The runner sets this automatically for
mixed stages. To re-plot manually:

```bash
python3 experiments/mvivf_ablation/scripts/plot.py \
  --results experiments/mvivf_ablation/results/fiqa/stage5 \
  --group-by num_spill --label-by index_name --prefix stage5
```
