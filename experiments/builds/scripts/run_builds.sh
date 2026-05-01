#!/usr/bin/env bash
# Build the indices for the main experiments.
#
# Usage:
#   experiments/builds/scripts/run_builds.sh --dataset beir5
#   experiments/builds/scripts/run_builds.sh --dataset arguana            # filters beir5.build.yaml
#   experiments/builds/scripts/run_builds.sh --dataset nq --method mvivf
#   experiments/builds/scripts/run_builds.sh --dataset vidore --method muvera
#   experiments/builds/scripts/run_builds.sh --dataset beir5 --method fastplaid
#
# Index binaries land at  results/indexes/<dataset>/<method>/<build_name>/index.bin .
# The configs reference that path; you may symlink results/indexes to scratch.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/experiments/builds/configs"
FILTER_PY="$REPO_ROOT/experiments/builds/scripts/filter_config.py"
cd "$REPO_ROOT"

DATASET=""
METHOD=""        # all | mvivf | muvera | vamana | svh_graph | fastplaid
TASK="build"     # build (only)
EXTRA_ARGS=()

# beir5 single-name shortcut: if --dataset is one of these, we filter beir5.build.yaml.
BEIR5_DATASETS=(nfcorpus scifact arguana scidocs fiqa)

# Aliases that map to a top-level config file.
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
    --task)    TASK="$2";    shift 2;;
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

# Pick the source YAML and any filter args.
SRC_YAML=""
FILTER_DATASET=""
if is_in "$DATASET" "${DATASET_ALIASES[@]}"; then
  SRC_YAML="$CONFIGS_DIR/${DATASET}.build.yaml"
elif is_in "$DATASET" "${BEIR5_DATASETS[@]}"; then
  SRC_YAML="$CONFIGS_DIR/beir5.build.yaml"
  FILTER_DATASET="$DATASET"
else
  echo "Unknown --dataset '$DATASET'. Aliases: ${DATASET_ALIASES[*]}; BEIR5 single-names: ${BEIR5_DATASETS[*]}." >&2
  exit 2
fi

if [[ ! -f "$SRC_YAML" ]]; then
  echo "[error] config not found: $SRC_YAML" >&2
  exit 2
fi

# Filter (dataset and/or method) into a temp YAML if needed.
NEED_FILTER=0
[[ -n "$FILTER_DATASET" ]] && NEED_FILTER=1
[[ -n "$METHOD" && "$METHOD" != "all" ]] && NEED_FILTER=1

# FastPlaid is BEIR5-only: if requested elsewhere, warn and skip.
if [[ "$METHOD" == "fastplaid" ]]; then
  if [[ "$DATASET" != "beir5" ]] && ! is_in "$DATASET" "${BEIR5_DATASETS[@]}"; then
    echo "[warn] FastPlaid is only configured for BEIR5; skipping." >&2
    exit 0
  fi
fi

CONFIG_PATH="$SRC_YAML"
if [[ "$NEED_FILTER" -eq 1 ]]; then
  TMP="$(mktemp "${TMPDIR:-/tmp}/main_build.XXXXXX.yaml")"
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

case "$TASK" in
  build)
    echo "=== Build: $CONFIG_PATH ==="
    python3 "$REPO_ROOT/benchmarks/benchmark_build.py" \
        --config "$CONFIG_PATH" \
        "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "Unknown --task '$TASK' (only 'build' is supported here; use the per-stage runners for search)." >&2
    exit 2
    ;;
esac
