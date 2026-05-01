#!/usr/bin/env bash
# Run the batch (search_all) throughput sweep.
#
# Reuses the search configs under experiments/latency/configs/ as the source
# of truth and only rewrites `results_dir` to land under
# experiments/batch/results/.
#
# Usage:
#   experiments/batch/scripts/run_batch.sh --dataset beir5
#   experiments/batch/scripts/run_batch.sh --dataset arguana --method mvivf
#   experiments/batch/scripts/run_batch.sh --dataset beir5 --method fastplaid

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/experiments/latency/configs"
FILTER_PY="$REPO_ROOT/experiments/builds/scripts/filter_config.py"
REWRITE_PY="$REPO_ROOT/experiments/builds/scripts/rewrite_results_dir.py"
cd "$REPO_ROOT"

DATASET=""
METHOD=""
EXTRA_ARGS=()

BEIR5_DATASETS=(nfcorpus scifact arguana scidocs fiqa)
DATASET_ALIASES=(beir5 nq hotpotqa nq500k quora vidore)

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
    --method)  METHOD="$2";  shift 2;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "Specify --dataset <alias|single-name>." >&2
  exit 2
fi

is_in() {
  local needle="$1"; shift
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

# FastPlaid is BEIR5-only.
if [[ "$METHOD" == "fastplaid" ]] \
   && [[ "$DATASET" != "beir5" ]] \
   && ! is_in "$DATASET" "${BEIR5_DATASETS[@]}"; then
  echo "[warn] FastPlaid is only configured for BEIR5; skipping." >&2
  exit 0
fi

SRC_YAML=""
FILTER_DATASET=""
if is_in "$DATASET" "${DATASET_ALIASES[@]}"; then
  SRC_YAML="$CONFIGS_DIR/${DATASET}.search.yaml"
elif is_in "$DATASET" "${BEIR5_DATASETS[@]}"; then
  SRC_YAML="$CONFIGS_DIR/beir5.search.yaml"
  FILTER_DATASET="$DATASET"
else
  echo "Unknown --dataset '$DATASET'." >&2
  exit 2
fi
[[ -f "$SRC_YAML" ]] || { echo "[error] config not found: $SRC_YAML" >&2; exit 2; }

# Rewrite results_dir from latency to batch.
REWRITTEN="$(mktemp "${TMPDIR:-/tmp}/main_batch_rw.XXXXXX.yaml")"
TEMP_YAMLS+=("$REWRITTEN")
python3 "$REWRITE_PY" --in "$SRC_YAML" --out "$REWRITTEN" \
    --from-stage latency --to-stage batch

# Optional dataset/method filtering.
NEED_FILTER=0
[[ -n "$FILTER_DATASET" ]] && NEED_FILTER=1
[[ -n "$METHOD" && "$METHOD" != "all" ]] && NEED_FILTER=1

CONFIG_PATH="$REWRITTEN"
if [[ "$NEED_FILTER" -eq 1 ]]; then
  TMP="$(mktemp "${TMPDIR:-/tmp}/main_batch_filt.XXXXXX.yaml")"
  TEMP_YAMLS+=("$TMP")
  args=(python3 "$FILTER_PY" --in "$REWRITTEN" --out "$TMP")
  [[ -n "$FILTER_DATASET" ]] && args+=(--dataset "$FILTER_DATASET")
  [[ -n "$METHOD" && "$METHOD" != "all" ]] && args+=(--method "$METHOD")
  if ! "${args[@]}"; then
    echo "[error] filter_config.py failed" >&2
    exit 2
  fi
  CONFIG_PATH="$TMP"
fi

echo "=== Batch: $CONFIG_PATH ==="
python3 "$REPO_ROOT/benchmarks/benchmark_search.py" \
    --config "$CONFIG_PATH" \
    --mode batch \
    "${EXTRA_ARGS[@]}"
