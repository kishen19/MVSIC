#!/usr/bin/env bash
# Run the MVIVF ablation sweeps (build + latency search).
#
# Usage:
#   scripts/run_ablation.sh --variant mvivf        # regular MVIVF
#   scripts/run_ablation.sh --variant mvivf_spill
#   scripts/run_ablation.sh --variant mvivf_flat
#   scripts/run_ablation.sh --variant all          # run all three
#   scripts/run_ablation.sh --variant mvivf --stage 1
#   scripts/run_ablation.sh --variant mvivf --stage 2 --dataset fiqa
#
#   With --dataset: prefers mvivf.stageN.<dataset>.{build,search}.yaml if present;
#   otherwise filters datasets: in mvivf.stageN.{build,search}.yaml; if that file
#   is missing, falls back to mvivf.{build,search}.yaml (same filter).
#   Filtered configs are written to a temp file and removed on exit.
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
DATASETS="arguana" # nq500k,fiqa
CONFIG_DATASET=""  # set by --dataset; selects *.stageN.<dataset>.*.yaml or filters multi-dataset YAML
STAGE="1"
GROUP_BY=""
WITH_WINNERS=0
EXTRA_ARGS=()

FILTER_PY="$REPO_ROOT/experiments/mvivf_ablation/scripts/filter_benchmark_config_dataset.py"
MVIVF_AB_CONFIG_PATH=""
TEMP_YAMLS=()

cleanup_mvivf_ablation_temps() {
  local f
  for f in "${TEMP_YAMLS[@]}"; do
    [[ -n "$f" && -f "$f" ]] && rm -f "$f"
  done
}
trap cleanup_mvivf_ablation_temps EXIT INT HUP TERM

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2;;
    --task)    TASK="$2";    shift 2;;
    --dataset) DATASETS="$2"; CONFIG_DATASET="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --stage)   STAGE="$2"; shift 2;;
    --group-by) GROUP_BY="$2"; shift 2;;
    --with-winners) WITH_WINNERS=1; shift;;
    *)         EXTRA_ARGS+=("$1"); shift;;
  esac
done

_register_filtered_config() {
  local src="$1"
  local ds="$2"
  local tmp
  tmp="$(mktemp "${TMPDIR:-/tmp}/mvivf_ablation.XXXXXX.yaml")"
  if ! python3 "$FILTER_PY" --in "$src" --dataset "$ds" --out "$tmp"; then
    rm -f "$tmp"
    return 1
  fi
  TEMP_YAMLS+=("$tmp")
  MVIVF_AB_CONFIG_PATH="$tmp"
}

# Sets global MVIVF_AB_CONFIG_PATH (never use inside command substitution).
config_resolve() {
  local name="$1"
  local kind="$2" # build|search
  local cfg=""
  MVIVF_AB_CONFIG_PATH=""

  if [[ -n "$CONFIG_DATASET" ]]; then
    if [[ "$CONFIG_DATASET" == *,* ]]; then
      echo "[error] --dataset accepts a single dataset name; use --datasets only for evaluation or generic multi-dataset configs." >&2
      return 2
    fi
  fi

  if [[ -n "$STAGE" ]]; then
    if [[ -n "$CONFIG_DATASET" ]]; then
      cfg="$CONFIGS_DIR/${name}.stage${STAGE}.${CONFIG_DATASET}.${kind}.yaml"
      if [[ -f "$cfg" ]]; then
        MVIVF_AB_CONFIG_PATH="$cfg"
        return 0
      fi
      cfg="$CONFIGS_DIR/${name}.stage${STAGE}.${kind}.yaml"
      if [[ -f "$cfg" ]]; then
        _register_filtered_config "$cfg" "$CONFIG_DATASET"
        return 0
      fi
      cfg="$CONFIGS_DIR/${name}.${kind}.yaml"
      if [[ -f "$cfg" ]]; then
        _register_filtered_config "$cfg" "$CONFIG_DATASET"
        return 0
      fi
      echo "[error] no config for --dataset ${CONFIG_DATASET}: tried" >&2
      echo "        $CONFIGS_DIR/${name}.stage${STAGE}.${CONFIG_DATASET}.${kind}.yaml" >&2
      echo "        $CONFIGS_DIR/${name}.stage${STAGE}.${kind}.yaml (filter)" >&2
      echo "        $CONFIGS_DIR/${name}.${kind}.yaml (filter)" >&2
      return 2
    fi
    cfg="$CONFIGS_DIR/${name}.stage${STAGE}.${kind}.yaml"
    if [[ -f "$cfg" ]]; then
      MVIVF_AB_CONFIG_PATH="$cfg"
      return 0
    fi
    echo "[warn] stage config not found: $cfg ; falling back to default ${name}.${kind}.yaml" >&2
  fi

  cfg="$CONFIGS_DIR/${name}.${kind}.yaml"
  if [[ -n "$CONFIG_DATASET" ]] && [[ -f "$cfg" ]]; then
    _register_filtered_config "$cfg" "$CONFIG_DATASET"
    return 0
  fi
  MVIVF_AB_CONFIG_PATH="$cfg"
}

default_group_by_for_stage() {
  local name="$1"
  local stage="$2"
  # MVIVF stage plan (winners feed forward into later stages):
  #   stage 1: k_per_level             (winner: 0)
  #   stage 2: max_leaf_size
  #   stage 3: max_depth
  #   stage 4: niters
  #   stage 5: s
  #   stage 6: max_point_clouds_per_cluster
  case "$name:$stage" in
    mvivf:1) echo "k_per_level" ;;
    mvivf:2) echo "max_leaf_size" ;;
    mvivf:3) echo "max_depth" ;;
    mvivf:4) echo "niters" ;;
    mvivf:5) echo "s" ;;
    mvivf:6) echo "max_point_clouds_per_cluster" ;;
    mvivf_spill:1) echo "num_spill" ;;
    mvivf_spill:2) echo "num_spill_l2" ;;
    mvivf_flat:1) echo "k_per_level" ;;
    *) echo "build_config" ;;
  esac
}

build_one() {
  local name="$1"
  config_resolve "$name" build || return "$?"
  echo "=== Build: $MVIVF_AB_CONFIG_PATH ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_build.py" \
      --config "$MVIVF_AB_CONFIG_PATH" \
      "${EXTRA_ARGS[@]}"
}

search_one() {
  local name="$1"
  config_resolve "$name" search || return "$?"
  echo "=== Search: $MVIVF_AB_CONFIG_PATH ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_search.py" \
      --config "$MVIVF_AB_CONFIG_PATH" \
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
    local indices_root_rel="results/mvivf_ablation"
    local results_root_rel="experiments/mvivf_ablation"
    local results_root="$REPO_ROOT/${results_root_rel}/results/${ds}/stage${STAGE}/${name}"
    local indices_root="$REPO_ROOT/${indices_root_rel}/indices/${ds}/${name}"
    echo "=== Evaluate: ${name} (${ds}) stage${STAGE} ==="
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
