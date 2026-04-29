# Codebook Training & Encoding Timings

Wall-clock time to (a) train each quantizer's codebook on a full .pcs file and (b) encode the same dataset.

## Reproducibility

- Command: `data-tools/measure_codebook_time.py --dataset fiqa:data/beir/fiqa/fiqa_points.pcs --dataset nq500k:data/beir/nq500k/nq500k_points.pcs --metric ip --out experiments/codebook_timings.md`
- Git SHA: `29ffdee`
- Host: `elm` (`Linux-5.15.0-171-generic-x86_64-with-glibc2.35`)
- Python: `3.11.7`
- Timestamp (UTC): `2026-04-21T20:05:41.894201`

## Results

| Dataset | Quantizer | Metric | N | Dim | Train (s) | Encode (s) |
|---------|-----------|--------|---|-----|-----------|------------|
| fiqa | pq | ip | 57638 | 128 | 0.137 | 0.225 |
| fiqa | fastscan | ip | 57638 | 128 | 0.065 | 0.067 |
| fiqa | rabitq | ip | 57638 | 128 | 0.021 | 0.268 |
| fiqa | turboquant | ip | 57638 | 128 | 0.000 | 0.315 |
| fiqa | onebittq | ip | 57638 | 128 | 0.000 | 0.126 |
| nq500k | pq | ip | 500000 | 128 | 0.125 | 2.018 |
| nq500k | fastscan | ip | 500000 | 128 | 0.062 | 0.497 |
| nq500k | rabitq | ip | 500000 | 128 | 0.036 | 1.715 |
| nq500k | turboquant | ip | 500000 | 128 | 0.000 | 1.985 |
| nq500k | onebittq | ip | 500000 | 128 | 0.000 | 0.705 |

## Observations

- All measured quantizers train a codebook in well under 1 second even on the
  nq500k dataset (500k point clouds, 128-d). Training is **not** a bottleneck.
- `turboquant` and `onebittq` report 0.000 s because they have no data-dependent
  codebook — encoding carries the full cost.
- The dominant load-time cost is **encoding** (largest here: `nq500k/pq` at
  ~2 s; everything else ≤ 2 s).
- `spqtq` crashed with SIGFPE (rc=-8) on both datasets, exiting inside the
  SPQTQ training path after the data was loaded. Likely an edge case
  (divide-by-zero) tied to the sample distribution. Tracked as a separate bug;
  does not affect this decision.

## Decision Rule

If any `train_sec > 60` for the largest dataset of interest, the codebook
should be persisted as a sidecar file (follow-up phase). Otherwise, the default
is to re-train on load inside `Quantizer::train(points)`.

**Decision:** Train-and-encode on load is the default for Phase 3. The sidecar
save path (`save_with_quantizer`, Phase 5) is kept as an optional optimization
but is **not** required for correctness or reasonable reload time.
