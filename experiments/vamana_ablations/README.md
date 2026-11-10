# Vamana ablations

The default `vamana.{build,search}.yaml` configs compare graph-building alpha values. The `vamana.quant.{build,search}.yaml` configs compare TQ 4-bit and TQ 1-bit leaf scoring on one raw graph, with `R=64`, build `L=128`, `alpha=1.0`, and `two_pass: false`.

The quantization search grid uses `k=10`, search `L` in `[64, 128, 256, 512, 1024, 2048, 4096]`, and rerank budgets `0`, `20`, and `40`. Both leaf variants enable TQ 8-bit reranking when the rerank budget is positive.

## Standard benchmark harness

On this machine, the extension loads under the `mvsic` Conda environment (Python 3.13); the default Python 3.12 has an ABI mismatch. Activate `mvsic` before using the standard Python harness. The dedicated native driver below does not import the extension.

From the repository root, select FiQA from the reusable nine-dataset configs:

```bash
experiments/vamana_ablations/scripts/run_ablation.sh --quant --dataset fiqa --task build
experiments/vamana_ablations/scripts/run_ablation.sh --quant --dataset fiqa --task search
experiments/vamana_ablations/scripts/run_ablation.sh --quant --dataset fiqa --task plot
```

Omit `--quant` for the alpha ablation, or omit `--dataset` to select all configured datasets. Use a separate invocation for each task when forwarding task-specific arguments such as `--query_subsample 0` or `--QPS_thresh 0` to search.

The build harness reuses an existing `results/vamana_ablations/indices/<dataset>/vamana/vamana_R64_a1.0/index.bin`. Search reloads this raw skeleton for each quantizer and writes to `experiments/vamana_ablations/results/<dataset>/vamana/vamana_R64_a1.0/<tq4|tq1>/k=10/`. Existing matching CSV rows are reused, including TQ 4-bit results produced by an earlier alpha ablation. The standard run therefore resumes prior work; it does not guarantee a fresh paired measurement of both variants.

By default, search deterministically samples at most 1,000 queries using seed 42 and stops each rerank sweep if QPS falls below 10, recall reaches 1.0, recall drops, or recall remains identical at the latest three steps after at least four steps. `--QPS_thresh 0` disables the QPS limit only; the recall rules still apply. `--query_subsample 0` uses all queries.

Quantization plots keep the two leaf variants separate, filter to the shared `vamana_R64_a1.0` build, and write `<dataset>_pareto_quant.pdf` under `experiments/vamana_ablations/results/_plots/`. They pool rerank budgets into each quantizer's Pareto frontier.

## Fresh paired FiQA measurements

The dedicated native driver `scripts/run_quant_native.py` runs the complete quantization grid into `experiments/vamana_ablations/quantization/fiqa/`. This separate output directory keeps a fresh paired comparison apart from the standard harness's cached alpha-ablation rows. Its `scripts/plot_quant.py` plotter reports the dedicated run.

```bash
# Compile the native harness if needed:
bazel build -c opt //mvsic/vamana:quant_bench
python3 experiments/vamana_ablations/scripts/run_quant_native.py --dataset fiqa \
  --binary bazel-bin/mvsic/vamana/quant_bench \
  --output-root experiments/vamana_ablations/quantization_reproduction
python3 experiments/vamana_ablations/scripts/plot_quant.py --dataset fiqa \
  --results-root experiments/vamana_ablations/quantization_reproduction
```

Build the Bazel target before running the driver. It defaults to the Bazel-built Vamana quantization benchmark when available; a native fallback at `quantization/bin/bench` can also be restored from the archive below. Both are Vamana-only executables. The reproduction output directory keeps new measurements separate from the archived November 5 run. The driver requires a new output root when the executable or other recorded settings change. The driver reads both quantization YAML files, checks their build parameters agree, builds or reloads the shared raw graph, and records its commands and settings. It measures every configured beam width without the standard harness's early stopping. Search uses one worker pinned to CPU 0, ten warmup queries per operating point, and one timed pass by default. FiQA has 648 queries, so all queries fit within the usual 1,000-query cap.

Both leaf variants share a serialized rotation, and every positive rerank budget shares one serialized TQ8 model. Training, corpus encoding, graph loading, and warmup occur outside query timing. Completed curves have hashes for the graph, models, query order, settings, and CSV; rerunning the driver reuses only matching completed curves. A `STOP` file in the dataset output directory terminates the active measurement.

Plots and measurement summaries are written under `quantization/fiqa/plots/`. The three-panel recall-versus-mean-latency figure compares bit depths separately at rerank budgets 0, 20, and 40. It uses measured points without interpolation; the target table marks unreached recalls explicitly.

The completed November 5, 2026 FiQA run contains all 42 operating points. Its report, raw measurements, plots, saved models, provenance, logs, validation records, and native binary were archived before local cleanup in [mvivf-fern-experiments-dump.zip](/ssd2/kishen/mvsic/mvivf-fern-experiments-dump.zip). The archive includes all experiment files and the FiQA graph at their original relative paths. Restore `experiments/vamana_ablations/quantization/` to inspect the run locally; large embedding datasets remain under `/ssd2/kishen/multivector/`. Historical commands and hashes remain as recorded, including source paths from the subsequently retired experimental harness.
