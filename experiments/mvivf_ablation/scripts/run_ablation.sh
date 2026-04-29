#!/usr/bin/env bash
# Run the MVIVF ablation sweeps (build + latency search).
#
# Usage:
#   scripts/run_ablation.sh --variant mvivf        # regular MVIVF
#   scripts/run_ablation.sh --variant mvivf_spill
#   scripts/run_ablation.sh --variant mvivf_flat
#   scripts/run_ablation.sh --variant all          # run all three
#   scripts/run_ablation.sh --variant mvivf --task build   # build only
#   scripts/run_ablation.sh --variant mvivf --task search  # search only
#
# Assumes the repo root is $PWD (or cd into it before running).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/experiments/mvivf_ablation/configs"

VARIANT="mvivf"
TASK="all"         # all | build | search
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2;;
    --task)    TASK="$2";    shift 2;;
    *)         EXTRA_ARGS+=("$1"); shift;;
  esac
done

build_one() {
  local name="$1"
  local cfg="$CONFIGS_DIR/${name}.build.yaml"
  echo "=== Build: $cfg ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_build.py" \
      --config "$cfg" \
      "${EXTRA_ARGS[@]}"
}

search_one() {
  local name="$1"
  local cfg="$CONFIGS_DIR/${name}.search.yaml"
  echo "=== Search: $cfg ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_search.py" \
      --config "$cfg" \
      --latency \
      "${EXTRA_ARGS[@]}"
}

run_one() {
  local name="$1"
  case "$TASK" in
    build)  build_one "$name";;
    search) search_one "$name";;
    all)    build_one "$name"; search_one "$name";;
    *) echo "unknown task: $TASK"; exit 2;;
  esac
}

case "$VARIANT" in
  mvivf|mvivf_spill|mvivf_flat) run_one "$VARIANT";;
  all)
    run_one mvivf
    run_one mvivf_spill
    run_one mvivf_flat
    ;;
  *)
    echo "unknown variant: $VARIANT"; exit 2;;
esac
