#!/usr/bin/env bash
# Run the leaf-quantization comparison (build raw mvivf skeleton if missing
# + per-quantizer latency sweeps + plot). Both variants (tq1, fs) ride on
# the same raw skeleton; FastScan is selected at search time via
# `quantizer: FastScan, block_size: 8`.
#
# Usage:
#   experiments/quantization_compare/scripts/run_quantization_compare.sh --dataset beir5
#   experiments/quantization_compare/scripts/run_quantization_compare.sh --dataset beirbig
#   experiments/quantization_compare/scripts/run_quantization_compare.sh --dataset nq
#   experiments/quantization_compare/scripts/run_quantization_compare.sh --dataset arguana --task plot
#   experiments/quantization_compare/scripts/run_quantization_compare.sh --dataset nq --quantizer tq1
#
# `--task <build|run|plot|all>` (default `all`):
#   build = build the raw mvivf skeleton (no-op if already built by
#           `experiments/builds/scripts/run_builds.sh`).
#   run   = run the latency sweep across all leaf quantizers.
#   plot  = re-render plots from existing CSVs (no build, no run).
#   all   = build + run + plot.
#
# `--quantizer <name[,name...]>` (optional) restricts which leaf-quantizer
# variants run (one of: tq1, fs). If omitted, both run.
#
# `--dataset <name>` (required for non-plot tasks):
#   * `beir5`     -> all five BEIR-5 shards (uses beir5.build/search.yaml)
#   * `beirbig`   -> all three BEIR-big shards (uses beirbig.build/search.yaml)
#   * <BEIR-5 shard>   -> one BEIR-5 shard, leaf=100 configs
#   * <BEIR-big shard> -> one BEIR-big shard, leaf=500 configs
#
# Single-thread BLAS / kernel pools are set so the latency reading is true
# per-query latency (matches experiments/latency).

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/experiments/quantization_compare/configs"
SCRIPTS_DIR="$REPO_ROOT/experiments/quantization_compare/scripts"
FILTER_PY="$REPO_ROOT/experiments/builds/scripts/filter_config.py"
cd "$REPO_ROOT"

DATASET=""
QUANTIZER=""
TASK="all"
EXTRA_ARGS=()

BEIR5_DATASETS=(nfcorpus scifact arguana scidocs fiqa)
BEIRBIG_DATASETS=(quora nq hotpotqa)
DATASET_ALIASES=(beir5 beirbig)

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
    --dataset)   DATASET="$2";   shift 2;;
    --quantizer) QUANTIZER="$2"; shift 2;;
    --task)      TASK="$2";      shift 2;;
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
  echo "Specify --dataset <name>. Aliases: ${DATASET_ALIASES[*]}; BEIR-5 shards: ${BEIR5_DATASETS[*]}; BEIR-big shards: ${BEIRBIG_DATASETS[*]}." >&2
  exit 2
fi

# Resolve which (build, search) yaml pair to use, plus an optional
# single-shard filter.
SRC_BUILD=""
SRC_SEARCH=""
FILTER_DATASET=""
if [[ -z "$DATASET" ]]; then
  : # plot-only with no dataset = render every results/<ds>/ dir
elif [[ "$DATASET" == "beir5" ]]; then
  SRC_BUILD="$CONFIGS_DIR/beir5.build.yaml"
  SRC_SEARCH="$CONFIGS_DIR/beir5.search.yaml"
elif [[ "$DATASET" == "beirbig" ]]; then
  SRC_BUILD="$CONFIGS_DIR/beirbig.build.yaml"
  SRC_SEARCH="$CONFIGS_DIR/beirbig.search.yaml"
elif is_in "$DATASET" "${BEIR5_DATASETS[@]}"; then
  SRC_BUILD="$CONFIGS_DIR/beir5.build.yaml"
  SRC_SEARCH="$CONFIGS_DIR/beir5.search.yaml"
  FILTER_DATASET="$DATASET"
elif is_in "$DATASET" "${BEIRBIG_DATASETS[@]}"; then
  SRC_BUILD="$CONFIGS_DIR/beirbig.build.yaml"
  SRC_SEARCH="$CONFIGS_DIR/beirbig.search.yaml"
  FILTER_DATASET="$DATASET"
else
  echo "Unknown --dataset '$DATASET'. Aliases: ${DATASET_ALIASES[*]}; BEIR-5 shards: ${BEIR5_DATASETS[*]}; BEIR-big shards: ${BEIRBIG_DATASETS[*]}." >&2
  exit 2
fi

filter_dataset() {
  # Filter a config down to a single dataset.
  local src="$1" out="$2"
  if ! python3 "$FILTER_PY" --in "$src" --out "$out" --dataset "$FILTER_DATASET"; then
    echo "[error] filter_config.py failed on $src" >&2
    return 1
  fi
}

filter_search_by_quantizer() {
  # filter_config.py doesn't support variant-level filtering; do it inline.
  local src="$1" out="$2"
  python3 - "$src" "$out" "$QUANTIZER" <<'PY'
import sys, yaml
src, out, q_csv = sys.argv[1:4]
wanted = {s.strip() for s in q_csv.split(",") if s.strip()}
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

# Materialize per-dataset views of the chosen yamls (only when filtering).
BUILD_YAML="$SRC_BUILD"
SEARCH_YAML="$SRC_SEARCH"
if [[ -n "$FILTER_DATASET" ]]; then
  TMP_BUILD="$(mktemp "${TMPDIR:-/tmp}/qcompare.build.XXXXXX.yaml")"
  TMP_SEARCH="$(mktemp "${TMPDIR:-/tmp}/qcompare.search.XXXXXX.yaml")"
  TEMP_YAMLS+=("$TMP_BUILD" "$TMP_SEARCH")
  filter_dataset "$SRC_BUILD"  "$TMP_BUILD"
  filter_dataset "$SRC_SEARCH" "$TMP_SEARCH"
  BUILD_YAML="$TMP_BUILD"
  SEARCH_YAML="$TMP_SEARCH"
fi

if [[ -n "$QUANTIZER" && -n "$SEARCH_YAML" ]]; then
  TMP_Q="$(mktemp "${TMPDIR:-/tmp}/qcompare.search_q.XXXXXX.yaml")"
  TEMP_YAMLS+=("$TMP_Q")
  filter_search_by_quantizer "$SEARCH_YAML" "$TMP_Q"
  SEARCH_YAML="$TMP_Q"
fi

if [[ "$TASK" == "build" || "$TASK" == "all" ]]; then
  if [[ -z "$BUILD_YAML" ]]; then
    echo "[error] --task build/all requires a --dataset (got none)." >&2
    exit 2
  fi
  echo "=== Build (raw mvivf skeleton): $BUILD_YAML ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_build.py" \
      --config "$BUILD_YAML" \
      "${EXTRA_ARGS[@]}"
fi

if [[ "$TASK" == "run" || "$TASK" == "all" ]]; then
  if [[ -z "$SEARCH_YAML" ]]; then
    echo "[error] --task run/all requires a --dataset (got none)." >&2
    exit 2
  fi
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
  if [[ -n "$FILTER_DATASET" ]]; then
    plot_args+=(--datasets "$FILTER_DATASET")
  elif [[ "$DATASET" == "beir5" ]]; then
    plot_args+=(--datasets "$(IFS=,; echo "${BEIR5_DATASETS[*]}")")
  elif [[ "$DATASET" == "beirbig" ]]; then
    plot_args+=(--datasets "$(IFS=,; echo "${BEIRBIG_DATASETS[*]}")")
  fi
  "${plot_args[@]}" || true
fi
