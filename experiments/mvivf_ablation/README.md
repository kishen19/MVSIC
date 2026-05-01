# MVIVF ablation runs

Default dataset in `run_ablation.sh` is `arguana`; override with `--dataset <name>` or `--datasets a,b` where needed.

Each stage uses YAML under `configs/` (`mvivf.stageN.*.yaml`). After we pick a winner for stage *N*, freeze those parameter values into the next stage’s configs before continuing.

---

## MVIVF

### Stage 1 — `k_per_level`

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 1 --dataset arguana --task all
```

### Stage 2 — `max_leaf_size`

(Update configs with the stage‑1 `k_per_level` winner.)

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 2 --dataset arguana --task all
```

### Stage 3 — `max_depth`

(Update configs with stage‑1 `k_per_level` and stage‑2 `max_leaf_size` winners.)

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 3 --dataset arguana --task all
```

### Stage 4 — `niters`

(Update configs with winners from stages 1–3.)

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 4 --dataset arguana --task all
```

### Stage 5 — `s`

(Update configs with winners from stages 1–4.)

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 5 --dataset arguana --task all
```

### Stage 6 — `max_point_clouds_per_cluster`

(Update configs with winners from stages 1–5.)

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf --stage 6 --dataset arguana --task all
```

---

## MVIVF Spill (after MVIVF stages)

Pin **all** MVIVF winners (`k_per_level`, `max_leaf_size`, `max_depth`, `niters`, `s`, `max_point_clouds_per_cluster`) in `configs/mvivf_spill.build.yaml` and `configs/mvivf_spill.search.yaml`, then run spill-only stages.

### Spill stage 1 — `num_spill`

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf_spill --stage 1 --dataset arguana --task all
```

### Spill stage 2 — `num_spill_l2`

(Update configs with the spill stage‑1 `num_spill` winner.)

```bash
experiments/mvivf_ablation/scripts/run_ablation.sh \
  --variant mvivf_spill --stage 2 --dataset arguana --task all
```

---

## MVIVF Flat (optional)

### Stage 1 — `k_per_level`

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
  --variant mvivf --stage 2 --dataset arguana --task evaluate
```

Optional: append `--with-winners` to pick budget winners during evaluate.

Indices default under `results/mvivf_ablation/indices/` per dataset entries in the YAMLs; search CSVs land under `experiments/mvivf_ablation/results/<dataset>/…`.
