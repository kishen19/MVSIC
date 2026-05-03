#!/usr/bin/env bash
# Build the indices for the main experiments.
#
# Usage (one dataset per invocation; use ``--method`` to pick one index or omit for all):
#   experiments/builds/scripts/run_builds.sh --dataset nfcorpus
#   experiments/builds/scripts/run_builds.sh --dataset arguana --method mvivf
#   experiments/builds/scripts/run_builds.sh --dataset vidore --method muvera
#   experiments/builds/scripts/run_builds.sh --dataset nfcorpus --method fastplaid
#
# FastPlaid is scoped to the classic BEIR-5 shards only (see fastplaid_scope.sh).
#
# Index binaries land at  results/indexes/<dataset>/<method>/<build_name>/index.bin .
# The configs reference that path; you may symlink results/indexes to scratch.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/experiments/builds/configs"
FILTER_PY="$REPO_ROOT/experiments/builds/scripts/filter_config.py"
# shellcheck disable=SC1091
source "$REPO_ROOT/experiments/builds/scripts/fastplaid_scope.sh"
cd "$REPO_ROOT"

DATASET=""
METHOD=""        # all | mvivf | muvera | vamana | svh_graph | fastplaid
TASK="build"     # build (only)
EXTRA_ARGS=()

# Names that share experiments/builds/configs/beir.build.yaml (must match BEIR_MERGED_NAMES in fastplaid_scope.sh).
BEIR_DATASETS=(nfcorpus scifact arguana scidocs fiqa quora nq hotpotqa nq500k)

# Top-level config files: vidore.build.yaml, msmarco.build.yaml
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
    --task)    TASK="$2";    shift 2;;
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

SRC_YAML=""
FILTER_DATASET=""
if is_in "$DATASET" "${DATASET_ALIASES[@]}"; then
  SRC_YAML="$CONFIGS_DIR/${DATASET}.build.yaml"
elif is_in "$DATASET" "${BEIR_DATASETS[@]}"; then
  SRC_YAML="$CONFIGS_DIR/beir.build.yaml"
  FILTER_DATASET="$DATASET"
else
  echo "Unknown --dataset '$DATASET'. Aliases: ${DATASET_ALIASES[*]}; BEIR names: ${BEIR_DATASETS[*]}." >&2
  exit 2
fi

if [[ ! -f "$SRC_YAML" ]]; then
  echo "[error] config not found: $SRC_YAML" >&2
  exit 2
fi

if fastplaid_skip_fastplaid_method "$DATASET" "${METHOD:-}"; then
  echo "[warn] FastPlaid builds only run on the classic BEIR-5 shards (nfcorpus … fiqa); skipping." >&2
  exit 0
fi

NEED_FILTER=0
[[ -n "$FILTER_DATASET" ]] && NEED_FILTER=1
[[ -n "${METHOD:-}" && "$METHOD" != "all" ]] && NEED_FILTER=1

CONFIG_PATH="$SRC_YAML"
if [[ "$NEED_FILTER" -eq 1 ]]; then
  TMP="$(mktemp "${TMPDIR:-/tmp}/main_build.XXXXXX.yaml")"
  TEMP_YAMLS+=("$TMP")
  args=(python3 "$FILTER_PY" --in "$SRC_YAML" --out "$TMP")
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
  strip_tmp="$(mktemp "${TMPDIR:-/tmp}/main_build_stripfp.XXXXXX.yaml")"
  TEMP_YAMLS+=("$strip_tmp")
  if ! python3 "$FILTER_PY" --in "$CONFIG_PATH" --out "$strip_tmp" --strip-indices fastplaid; then
    echo "[error] filter_config.py --strip-indices failed" >&2
    exit 2
  fi
  CONFIG_PATH="$strip_tmp"
fi

drop_caches_tail() {
  sync || true
  echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
}

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

drop_caches_tail
