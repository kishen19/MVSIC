#!/usr/bin/env bash
# Run Vamana ablations (single stage).
#
# Usage:
#   experiments/vamana_ablations/scripts/run_ablation.sh
#   experiments/vamana_ablations/scripts/run_ablation.sh --task build
#   experiments/vamana_ablations/scripts/run_ablation.sh --task search
#   experiments/vamana_ablations/scripts/run_ablation.sh --task plot
#   experiments/vamana_ablations/scripts/run_ablation.sh --dataset fiqa
#
# Build sweep:
#   R in {64, 200}, alpha in {1.0, 1.2}
# Search:
#   TQ4 variant, L sweep, num_rerank sweep.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/experiments/vamana_ablations/configs"
FILTER_PY="$REPO_ROOT/experiments/builds/scripts/filter_config.py"

TASK="all"           # all | build | search | plot
CONFIG_DATASET=""    # optional single dataset filter
EXTRA_ARGS=()
TEMP_YAMLS=()
AB_CONFIG_PATH=""

cleanup_vamana_ablation_temps() {
  local f
  for f in "${TEMP_YAMLS[@]}"; do
    [[ -n "$f" && -f "$f" ]] && rm -f "$f"
  done
}
trap cleanup_vamana_ablation_temps EXIT INT HUP TERM

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)    TASK="$2"; shift 2;;
    --dataset) CONFIG_DATASET="$2"; shift 2;;
    *)         EXTRA_ARGS+=("$1"); shift;;
  esac
done

if [[ -n "$CONFIG_DATASET" && "$CONFIG_DATASET" == *,* ]]; then
  echo "[error] --dataset accepts a single dataset name." >&2
  exit 2
fi

_register_filtered_config() {
  local src="$1"
  local ds="$2"
  local tmp
  tmp="$(mktemp "${TMPDIR:-/tmp}/vamana_ablation.XXXXXX.yaml")"
  if ! python3 "$FILTER_PY" --in "$src" --dataset "$ds" --method vamana --out "$tmp"; then
    rm -f "$tmp"
    return 1
  fi
  TEMP_YAMLS+=("$tmp")
  AB_CONFIG_PATH="$tmp"
}

config_resolve() {
  local kind="$1" # build|search
  local cfg="$CONFIGS_DIR/vamana.${kind}.yaml"
  AB_CONFIG_PATH=""
  if [[ ! -f "$cfg" ]]; then
    echo "[error] missing config: $cfg" >&2
    return 2
  fi
  if [[ -n "$CONFIG_DATASET" ]]; then
    _register_filtered_config "$cfg" "$CONFIG_DATASET"
    return 0
  fi
  AB_CONFIG_PATH="$cfg"
}

build_run() {
  config_resolve build || return "$?"
  echo "=== Build: $AB_CONFIG_PATH ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_build.py" \
      --config "$AB_CONFIG_PATH" \
      "${EXTRA_ARGS[@]}"
}

search_run() {
  config_resolve search || return "$?"
  echo "=== Search (latency): $AB_CONFIG_PATH ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_search.py" \
      --config "$AB_CONFIG_PATH" \
      --mode latency \
      "${EXTRA_ARGS[@]}"
}

plot_run() {
  local plot_py="$REPO_ROOT/experiments/vamana_ablations/scripts/plot.py"
  if [[ ! -f "$plot_py" ]]; then
    echo "[error] missing plot script: $plot_py" >&2
    return 2
  fi
  if [[ -n "$CONFIG_DATASET" ]]; then
    python3 "$plot_py" --datasets "$CONFIG_DATASET" "${EXTRA_ARGS[@]}"
  else
    python3 "$plot_py" "${EXTRA_ARGS[@]}"
  fi
}

case "$TASK" in
  build)  build_run;;
  search) search_run;;
  plot)   plot_run;;
  all)    build_run; search_run; plot_run;;
  *)      echo "unknown task: $TASK"; exit 2;;
esac
