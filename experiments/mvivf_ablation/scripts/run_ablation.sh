#!/usr/bin/env bash
# Run the MVIVF ablation sweeps (build + latency search).
#
# Usage:
#   scripts/run_ablation.sh --variant mvivf        # regular MVIVF
#   scripts/run_ablation.sh --variant mvivf_spill
#   scripts/run_ablation.sh --variant mvivf_flat
#   scripts/run_ablation.sh --variant all          # run all three
#   scripts/run_ablation.sh --variant mvivf --stage 1
#   scripts/run_ablation.sh --variant mvivf --task build   # build only
#   scripts/run_ablation.sh --variant mvivf --task search  # search only
#   scripts/run_ablation.sh --variant mvivf --task evaluate
#
# Assumes the repo root is $PWD (or cd into it before running).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/experiments/mvivf_ablation/configs"

VARIANT="mvivf"
TASK="all"         # all | build | search | evaluate
DATASETS="nq500k,fiqa"
STAGE=""
GROUP_BY=""
WITH_WINNERS=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2;;
    --task)    TASK="$2";    shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --stage)   STAGE="$2"; shift 2;;
    --group-by) GROUP_BY="$2"; shift 2;;
    --with-winners) WITH_WINNERS=1; shift;;
    *)         EXTRA_ARGS+=("$1"); shift;;
  esac
done

config_path() {
  local name="$1"
  local kind="$2" # build|search
  local cfg=""
  if [[ -n "$STAGE" ]]; then
    cfg="$CONFIGS_DIR/${name}.stage${STAGE}.${kind}.yaml"
    if [[ -f "$cfg" ]]; then
      echo "$cfg"
      return 0
    fi
    echo "[warn] stage config not found: $cfg ; falling back to default ${name}.${kind}.yaml" >&2
  fi
  echo "$CONFIGS_DIR/${name}.${kind}.yaml"
}

default_group_by_for_stage() {
  local name="$1"
  local stage="$2"
  case "$name:$stage" in
    mvivf:1) echo "k_per_level" ;;
    mvivf:2) echo "max_leaf_size" ;;
    mvivf:3) echo "max_depth" ;;
    mvivf:4) echo "niters" ;;
    mvivf:5) echo "build_config" ;;  # grid stage (mpcc/mpcik)
    mvivf:6) echo "s" ;;
    mvivf_spill:1) echo "num_spill" ;;
    mvivf_spill:2) echo "num_spill_l2" ;;
    mvivf_flat:1) echo "k_per_level" ;;
    *) echo "build_config" ;;
  esac
}

build_one() {
  local name="$1"
  local cfg
  cfg="$(config_path "$name" build)"
  echo "=== Build: $cfg ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_build.py" \
      --config "$cfg" \
      "${EXTRA_ARGS[@]}"
}

search_one() {
  local name="$1"
  local cfg
  cfg="$(config_path "$name" search)"
  echo "=== Search: $cfg ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_search.py" \
      --config "$cfg" \
      --latency \
      "${EXTRA_ARGS[@]}"
}

evaluate_one() {
  local name="$1"
  local summary_py="$REPO_ROOT/experiments/mvivf_ablation/scripts/stage_summary.py"
  local stage_label="stage${STAGE:-1}"
  local group="$GROUP_BY"
  if [[ -z "$group" ]]; then
    if [[ -n "$STAGE" ]]; then
      group="$(default_group_by_for_stage "$name" "$STAGE")"
    else
      group="build_config"
    fi
  fi
  IFS=',' read -r -a ds_arr <<< "$DATASETS"
  for ds in "${ds_arr[@]}"; do
    ds="$(echo "$ds" | xargs)"
    [[ -z "$ds" ]] && continue
    local base="data/beir/${ds}/${ds}"
    local indices_root_rel="results/mvivf_ablations"
    local results_root_rel="experiments/mvivf_ablation"
    local results_root="$REPO_ROOT/${results_root_rel}/results/${ds}/${name}"
    local indices_root="$REPO_ROOT/${indices_root_rel}/indices/${ds}/${name}"
    echo "=== Evaluate: ${name} (${ds}) ==="
    local cmd=(
      python3 "$summary_py"
      --results "$results_root"
      --indices-root "$indices_root"
      --db "$REPO_ROOT/${base}_points.pcs"
      --queries "$REPO_ROOT/${base}_queries.pcs"
      --gt "$REPO_ROOT/${base}_chamfer_neighbors.gt"
      --group-by "$group"
      --stage-label "$stage_label"
    )
    if [[ "$WITH_WINNERS" -eq 1 ]]; then
      cmd+=(--with-budget-winners)
    fi
    "${cmd[@]}" \
        "${EXTRA_ARGS[@]}"
  done
}

run_one() {
  local name="$1"
  case "$TASK" in
    build)  build_one "$name";;
    search) search_one "$name";;
    evaluate) evaluate_one "$name";;
    all)    build_one "$name"; search_one "$name"; evaluate_one "$name";;
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
