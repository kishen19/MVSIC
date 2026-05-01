#!/usr/bin/env bash
# Run the single-thread per-query latency sweep.
#
# Usage:
#   experiments/latency/scripts/run_latency.sh --dataset beir5
#   experiments/latency/scripts/run_latency.sh --dataset arguana --method mvivf
#   experiments/latency/scripts/run_latency.sh --dataset vidore --method muvera
#   experiments/latency/scripts/run_latency.sh --dataset beir5 --method fastplaid

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/experiments/latency/configs"
FILTER_PY="$REPO_ROOT/experiments/builds/scripts/filter_config.py"
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
  echo "Specify --dataset <alias|single-name>. Aliases: ${DATASET_ALIASES[*]}; or one of beir5: ${BEIR5_DATASETS[*]}." >&2
  exit 2
fi

is_in() {
  local needle="$1"; shift
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

SRC_YAML=""
FILTER_DATASET=""
if is_in "$DATASET" "${DATASET_ALIASES[@]}"; then
  SRC_YAML="$CONFIGS_DIR/${DATASET}.search.yaml"
elif is_in "$DATASET" "${BEIR5_DATASETS[@]}"; then
  SRC_YAML="$CONFIGS_DIR/beir5.search.yaml"
  FILTER_DATASET="$DATASET"
else
  echo "Unknown --dataset '$DATASET'. Aliases: ${DATASET_ALIASES[*]}; BEIR5 single-names: ${BEIR5_DATASETS[*]}." >&2
  exit 2
fi

[[ -f "$SRC_YAML" ]] || { echo "[error] config not found: $SRC_YAML" >&2; exit 2; }

# FastPlaid is BEIR5-only.
if [[ "$METHOD" == "fastplaid" ]] \
   && [[ "$DATASET" != "beir5" ]] \
   && ! is_in "$DATASET" "${BEIR5_DATASETS[@]}"; then
  echo "[warn] FastPlaid is only configured for BEIR5; skipping." >&2
  exit 0
fi

NEED_FILTER=0
[[ -n "$FILTER_DATASET" ]] && NEED_FILTER=1
[[ -n "$METHOD" && "$METHOD" != "all" ]] && NEED_FILTER=1

CONFIG_PATH="$SRC_YAML"
if [[ "$NEED_FILTER" -eq 1 ]]; then
  TMP="$(mktemp "${TMPDIR:-/tmp}/main_latency.XXXXXX.yaml")"
  TEMP_YAMLS+=("$TMP")
  args=(python3 "$FILTER_PY" --in "$SRC_YAML" --out "$TMP")
  [[ -n "$FILTER_DATASET" ]] && args+=(--dataset "$FILTER_DATASET")
  [[ -n "$METHOD" && "$METHOD" != "all" ]] && args+=(--method "$METHOD")
  if ! "${args[@]}"; then
    echo "[error] filter_config.py failed" >&2
    exit 2
  fi
  CONFIG_PATH="$TMP"
fi

echo "=== Latency: $CONFIG_PATH ==="
python3 "$REPO_ROOT/benchmarks/benchmark_search.py" \
    --config "$CONFIG_PATH" \
    --mode latency \
    "${EXTRA_ARGS[@]}"
