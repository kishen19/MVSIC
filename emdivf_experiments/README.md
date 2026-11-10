# EMDIVF exploratory work

The runnable method is in [`mvsic/emdivf`](../mvsic/emdivf/README.md).
This directory contains selected exploratory tools, separate from the paper
experiments in `experiments/`.

Read **[EXPLORATION.md](EXPLORATION.md)** for the complete findings, important
numbers, plots, unsuccessful approaches, and proposals that have not been run.
The checked-in `figures/` are selected snapshots of existing results; no new
benchmark measurements were made during the cleanup.

## Layout

```text
mvsic/emdivf/              C++ method, builder, graph/search CLI, EMD tests
emdivf_experiments/
  EXPLORATION.md           consolidated research record
  figures/                selected portable result plots
  embedding/              FiQA/HotpotQA query and corpus-prototype preparation
  diagnostics/            exact routing/probe evaluation tools
  build_ablation/         parent-pool and document-first construction diagnostics
  doc_ann_pilot/          document-index candidate-generator diagnostics
  quantization_audit/     analysis of the saved quantization investigation
  prefix_build/           proposed prefix-doubling router/plotting draft
  artifacts -> /ssd/...   local results, indexes, caches, and original trial code
  results/                regenerated outputs (ignored)
```

`prefix_build/` is an unfinished prototype. There is no integrated
prefix-doubling bucket builder or measured result for it yet.

## Local artifact policy

Commit source, small configuration files, the report, and selected figures.
Keep datasets, encodings, indexes, checkpoints, logs, and bulk result grids on
SSD. The exact path `emdivf_experiments/artifacts` is ignored, including when
it is a symlink. Code used by a Bazel target must be a real repository file;
no build target depends on this local link.

On this machine the link points to the preserved `/ssd2/kishen/emdivf` tree.
On another machine, create the link to your own artifact directory:

```bash
ln -s /path/to/emdivf-artifacts emdivf_experiments/artifacts
```

The local tree retains `results/`, `indexes/`, the original experiment
scripts, and `archive/repo_exploratory/`. That archive preserves the original
exploratory directories moved out of `experiments/`. Historical scripts in
the artifact tree may still contain their original absolute paths; use the
repository tools for maintained entry points. No old measurements were deleted.

## Useful commands

Follow the method's [build/search instructions](../mvsic/emdivf/README.md)
for a fresh run. Dataset preparation reuses the existing `embed` package:

```bash
python -m emdivf_experiments.embedding.fiqa_splits \
  --source /path/to/extracted/fiqa --output data/beir/fiqa
python -m emdivf_experiments.embedding.verify_fiqa_test_ids \
  --source /path/to/extracted/fiqa --output data/beir/fiqa
python -m emdivf_experiments.embedding.hotpotqa_splits \
  --output data/beir/hotpotqa
python -m emdivf_experiments.embedding.corpus_prototypes --help
```

The encoding helpers refuse to overwrite existing outputs. Split preparation
can use relevance judgments to identify valid source queries; retrieval
evaluation uses exhaustive Chamfer neighbors as ground truth.

Plotting saved results does not invoke retrieval. For the retained FiQA
latency comparison and HotpotQA work comparison:

```bash
python emdivf_experiments/plot_saved_results.py --dataset both --include-copied
```

The output goes to ignored `emdivf_experiments/results/`. Pass `--artifacts`
to read a different local result tree. The C++ tools in `diagnostics/` measure
coverage under exhaustive exact-EMD prototype routing; their work curves
exclude that routing cost.

The cleanup was reviewed statically. Builds, tests, encoding, and benchmarks
were deliberately left unrun while the shared machine was busy. The saved
results describe the earlier experimental implementation, not a fresh timing
of the relocated method.
