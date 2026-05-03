#!/usr/bin/env bash
# Run the batch (search_all) throughput sweep.
#
# Reuses the search configs under experiments/latency/configs/ as the source
# of truth and only rewrites `results_dir` to land under
# experiments/batch/results/.
#
# Usage (one dataset per invocation):
#   experiments/batch/scripts/run_batch.sh --dataset nfcorpus
#   experiments/batch/scripts/run_batch.sh --dataset arguana --method mvivf
#
# FastPlaid is scoped to the classic BEIR-5 shards only (see fastplaid_scope.sh).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/experiments/latency/configs"
FILTER_PY="$REPO_ROOT/experiments/builds/scripts/filter_config.py"
REWRITE_PY="$REPO_ROOT/experiments/builds/scripts/rewrite_results_dir.py"
# shellcheck disable=SC1091
source "$REPO_ROOT/experiments/builds/scripts/fastplaid_scope.sh"
cd "$REPO_ROOT"

DATASET=""
METHOD=""
EXTRA_ARGS=()

BEIR_DATASETS=(nfcorpus scifact arguana scidocs fiqa quora nq hotpotqa nq500k)
DATASET_ALIASES=(vidore msmarco)

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
  echo "Specify --dataset <name>. Aliases: ${DATASET_ALIASES[*]}; or one BEIR dataset: ${BEIR_DATASETS[*]}." >&2
  exit 2
fi

is_in() {
  local needle="$1"; shift
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

if fastplaid_skip_fastplaid_method "$DATASET" "${METHOD:-}"; then
  echo "[warn] FastPlaid batch search only runs on the classic BEIR-5 shards (nfcorpus … fiqa); skipping." >&2
  exit 0
fi

SRC_YAML=""
FILTER_DATASET=""
if is_in "$DATASET" "${DATASET_ALIASES[@]}"; then
  SRC_YAML="$CONFIGS_DIR/${DATASET}.search.yaml"
elif is_in "$DATASET" "${BEIR_DATASETS[@]}"; then
  SRC_YAML="$CONFIGS_DIR/beir.search.yaml"
  FILTER_DATASET="$DATASET"
else
  echo "Unknown --dataset '$DATASET'. Aliases: ${DATASET_ALIASES[*]}; BEIR names: ${BEIR_DATASETS[*]}." >&2
  exit 2
fi
[[ -f "$SRC_YAML" ]] || { echo "[error] config not found: $SRC_YAML" >&2; exit 2; }

REWRITTEN="$(mktemp "${TMPDIR:-/tmp}/main_batch_rw.XXXXXX.yaml")"
TEMP_YAMLS+=("$REWRITTEN")
python3 "$REWRITE_PY" --in "$SRC_YAML" --out "$REWRITTEN" \
    --from-stage latency --to-stage batch

NEED_FILTER=0
[[ -n "$FILTER_DATASET" ]] && NEED_FILTER=1
[[ -n "${METHOD:-}" && "$METHOD" != "all" ]] && NEED_FILTER=1

CONFIG_PATH="$REWRITTEN"
if [[ "$NEED_FILTER" -eq 1 ]]; then
  TMP="$(mktemp "${TMPDIR:-/tmp}/main_batch_filt.XXXXXX.yaml")"
  TEMP_YAMLS+=("$TMP")
  args=(python3 "$FILTER_PY" --in "$REWRITTEN" --out "$TMP")
  [[ -n "$FILTER_DATASET" ]] && args+=(--dataset "$FILTER_DATASET")
  [[ -n "${METHOD:-}" && "$METHOD" != "all" ]] && args+=(--method "$METHOD")
  if ! "${args[@]}"; then
    echo "[error] filter_config.py failed" >&2
    exit 2
  fi
  CONFIG_PATH="$TMP"
fi

eff_ds="${FILTER_DATASET:-}"
if [[ -z "$eff_ds" && "$DATASET" == "msmarco" ]]; then
  eff_ds="msmarco"
fi
if [[ -n "$eff_ds" ]] && fastplaid_should_strip_indices "$eff_ds"; then
  strip_tmp="$(mktemp "${TMPDIR:-/tmp}/main_batch_stripfp.XXXXXX.yaml")"
  TEMP_YAMLS+=("$strip_tmp")
  if ! python3 "$FILTER_PY" --in "$CONFIG_PATH" --out "$strip_tmp" --strip-indices fastplaid; then
    echo "[error] filter_config.py --strip-indices failed" >&2
    exit 2
  fi
  CONFIG_PATH="$strip_tmp"
fi

echo "=== Batch: $CONFIG_PATH ==="
python3 "$REPO_ROOT/benchmarks/benchmark_search.py" \
    --config "$CONFIG_PATH" \
    --mode batch \
    "${EXTRA_ARGS[@]}"

sync || true
echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
