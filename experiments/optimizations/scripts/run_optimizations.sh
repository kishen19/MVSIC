#!/usr/bin/env bash
# Run the MVIVF optimizations ladder (build skeletons + per-stage latency
# sweeps, optionally re-rendering the Pareto and speedup plots).
#
# Usage:
#   experiments/optimizations/scripts/run_optimizations.sh --dataset nq
#   experiments/optimizations/scripts/run_optimizations.sh --dataset arguana
#   experiments/optimizations/scripts/run_optimizations.sh --dataset nq --task plot
#   experiments/optimizations/scripts/run_optimizations.sh --dataset nq --stage s2_leaf_tq1bit
#   experiments/optimizations/scripts/run_optimizations.sh --dataset all
#
# `--task <build|run|plot|all>` (default `all`):
#   build = build the mvivf skeleton only.
#   run   = run the per-stage latency sweep (skips build if indices exist).
#   plot  = re-render plots from existing CSVs (no build, no run).
#   all   = build + run + plot.
#
# `--stage <variant_name[,variant_name...]>` (optional) restricts which
# stages run (e.g. s0_baseline,s1_leaf_tq1bit). If omitted, all five run.
#
# `--dataset <name|all>` (required for non-plot tasks): one of the BEIR-5
# shards (nfcorpus, scifact, arguana, scidocs, fiqa), BEIR-big shards
# (quora, nq, hotpotqa), or `nq500k`. `all` runs every dataset declared in
# the config.
#
# Single-thread BLAS / kernel pools are set so the latency reading is true
# per-query latency (matching the main latency experiments).
#
# Query subsampling: search runs delegate to benchmarks/benchmark_search.py
# which defaults to --query_subsample 1000 --query_subsample_seed 42 (so
# datasets with >1000 queries use the same 1000 across runs and stages).
# Override per call by appending e.g. `--query_subsample 0` (all queries).

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/experiments/optimizations/configs"
SCRIPTS_DIR="$REPO_ROOT/experiments/optimizations/scripts"
FILTER_PY="$REPO_ROOT/experiments/builds/scripts/filter_config.py"
cd "$REPO_ROOT"

DATASET=""
STAGE=""        # comma-separated variant names; empty = all stages
TASK="all"      # build | run | plot | all
EXTRA_ARGS=()

ALL_DATASETS=(nfcorpus scifact arguana scidocs fiqa quora nq hotpotqa nq500k)

TEMP_YAMLS=()
cleanup_tmp_yamls() {
  local f
  for f in "${TEMP_YAMLS[@]}"; do
    [[ -n "$f" && -f "$f" ]] && rm -f "$f"
  done
}
trap cleanup_tmp_yamls EXIT INT HUP TERM

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --stage)   STAGE="$2";   shift 2;;
    --task)    TASK="$2";    shift 2;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

case "$TASK" in
  build|run|plot|all) ;;
  *) echo "Unknown --task '$TASK' (use build|run|plot|all)" >&2; exit 2;;
esac

is_in() {
  local needle="$1"; shift
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

if [[ -z "$DATASET" && "$TASK" != "plot" ]]; then
  echo "Specify --dataset <name|all>. Available: ${ALL_DATASETS[*]} (or 'all')." >&2
  exit 2
fi

if [[ -n "$DATASET" && "$DATASET" != "all" ]]; then
  if ! is_in "$DATASET" "${ALL_DATASETS[@]}"; then
    echo "Unknown --dataset '$DATASET'. Available: ${ALL_DATASETS[*]} (or 'all')." >&2
    exit 2
  fi
fi

# ---- Filter the source YAMLs by dataset (and optionally by stage). -------
filter_yaml() {
  local src="$1" out="$2"
  local args=(python3 "$FILTER_PY" --in "$src" --out "$out")
  [[ -n "$DATASET" && "$DATASET" != "all" ]] && args+=(--dataset "$DATASET")
  if ! "${args[@]}"; then
    echo "[error] filter_config.py failed on $src" >&2
    return 1
  fi
}

filter_search_by_stage() {
  # Drop builds[].variants[] entries whose name isn't in the user's stage list.
  # filter_config.py doesn't support variant-level filtering, so we use a
  # small inline yaml munger.
  local src="$1" out="$2"
  python3 - "$src" "$out" "$STAGE" <<'PY'
import sys, yaml

src, out, stage_csv = sys.argv[1:4]
wanted = {s.strip() for s in stage_csv.split(",") if s.strip()}
with open(src) as f:
    cfg = yaml.safe_load(f)
for index in cfg.get("indices", []) or []:
    new_builds = []
    for build in index.get("builds", []) or []:
        variants = [v for v in (build.get("variants") or []) if v.get("name") in wanted]
        if not variants:
            continue
        b = dict(build)
        b["variants"] = variants
        new_builds.append(b)
    index["builds"] = new_builds
cfg["indices"] = [i for i in cfg.get("indices", []) if i.get("builds")]
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

BUILD_YAML="$CONFIGS_DIR/optimizations.build.yaml"
SEARCH_YAML="$CONFIGS_DIR/optimizations.search.yaml"

if [[ -n "$DATASET" && "$DATASET" != "all" ]]; then
  TMP_BUILD="$(mktemp "${TMPDIR:-/tmp}/optimizations.build.XXXXXX.yaml")"
  TMP_SEARCH="$(mktemp "${TMPDIR:-/tmp}/optimizations.search.XXXXXX.yaml")"
  TEMP_YAMLS+=("$TMP_BUILD" "$TMP_SEARCH")
  filter_yaml "$BUILD_YAML"  "$TMP_BUILD"
  filter_yaml "$SEARCH_YAML" "$TMP_SEARCH"
  BUILD_YAML="$TMP_BUILD"
  SEARCH_YAML="$TMP_SEARCH"
fi

if [[ -n "$STAGE" ]]; then
  TMP_STAGE="$(mktemp "${TMPDIR:-/tmp}/optimizations.search_stage.XXXXXX.yaml")"
  TEMP_YAMLS+=("$TMP_STAGE")
  filter_search_by_stage "$SEARCH_YAML" "$TMP_STAGE"
  SEARCH_YAML="$TMP_STAGE"
fi

# ---- Tasks ----------------------------------------------------------------

if [[ "$TASK" == "build" || "$TASK" == "all" ]]; then
  echo "=== Build: $BUILD_YAML ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_build.py" \
      --config "$BUILD_YAML" \
      "${EXTRA_ARGS[@]}"
fi

if [[ "$TASK" == "run" || "$TASK" == "all" ]]; then
  echo "=== Latency: $SEARCH_YAML ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_search.py" \
      --config "$SEARCH_YAML" \
      --mode latency \
      "${EXTRA_ARGS[@]}"

  sync || true
  echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
fi

if [[ "$TASK" == "plot" || "$TASK" == "all" ]]; then
  echo "=== Plots ==="
  plot_args=(python3 "$SCRIPTS_DIR/plot.py")
  if [[ -n "$DATASET" && "$DATASET" != "all" ]]; then
    plot_args+=(--datasets "$DATASET")
  fi
  "${plot_args[@]}" || true
fi
