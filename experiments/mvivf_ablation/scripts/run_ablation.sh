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
# Query subsampling: search runs delegate to benchmarks/benchmark_search.py,
# which defaults to --query_subsample 1000 --query_subsample_seed 42 (so for
# any dataset with >1000 queries, the same 1000 are used across runs and
# modes). Override per call by appending e.g. ``--query_subsample 0`` (all
# queries) or ``--query_subsample 5000`` to the runner command.
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
  #   stage 5: num_spill (mvivf vs mvivf_spill mixed)
  #   stage 6: query_compression (None vs Wards, mixed mvivf + mvivf_spill)
  #   stage 7: variant_name (1BTQ vs FastScan quantizer variants)
  case "$name:$stage" in
    mvivf:1) echo "k_per_level" ;;
    mvivf:2) echo "max_leaf_size" ;;
    mvivf:3) echo "max_depth" ;;
    mvivf:4) echo "niters" ;;
    mvivf:5) echo "num_spill" ;;
    mvivf:6) echo "query_compression" ;;
    mvivf:7) echo "variant_name" ;;
    mvivf_spill:1) echo "num_spill" ;;
    mvivf_spill:2) echo "num_spill_l2" ;;
    mvivf_flat:1) echo "k_per_level" ;;
    *) echo "build_config" ;;
  esac
}

# Stages 5/6/7 mix multiple index families (mvivf + mvivf_spill) under the
# same stage results dir, so we additionally split each curve by `index_name`.
default_label_by_for_stage() {
  local name="$1"
  local stage="$2"
  case "$name:$stage" in
    mvivf:5|mvivf:6|mvivf:7) echo "index_name" ;;
    *) echo "" ;;
  esac
}

# Whether the stage results live directly under stage<N>/ (true for mixed-
# method stages where evaluate_one must point --results at the stage dir
# rather than stage<N>/<name>/).
stage_is_mixed() {
  local name="$1"
  local stage="$2"
  case "$name:$stage" in
    mvivf:5|mvivf:6|mvivf:7) return 0 ;;
    *) return 1 ;;
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
  local label_by="$(default_label_by_for_stage "$name" "$STAGE")"
  IFS=',' read -r -a ds_arr <<< "$DATASETS"
  for ds in "${ds_arr[@]}"; do
    ds="$(echo "$ds" | xargs)"
    [[ -z "$ds" ]] && continue
    local base="data/beir/${ds}/${ds}"
    local indices_root_rel="results/mvivf_ablation"
    local results_root_rel="experiments/mvivf_ablation"
    local results_root
    if stage_is_mixed "$name" "$STAGE"; then
      # Mixed-method stage: load every index_name subtree under stage<N>/.
      results_root="$REPO_ROOT/${results_root_rel}/results/${ds}/stage${STAGE}"
    else
      results_root="$REPO_ROOT/${results_root_rel}/results/${ds}/stage${STAGE}/${name}"
    fi
    local indices_root="$REPO_ROOT/${indices_root_rel}/indices/${ds}/${name}"
    # Stage PDFs go to the canonical _plots/ tree; we tag them with
    # (dataset, variant, stage) so runs share the directory cleanly.
    local plots_out_dir="$REPO_ROOT/${results_root_rel}/results/_plots"
    local plots_prefix="${ds}_${name}_${stage_label}"
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
      --plots-out-dir "$plots_out_dir"
      --plots-prefix "$plots_prefix"
    )
    if [[ -n "$label_by" ]]; then
      cmd+=(--label-by "$label_by")
    fi
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

# Best-effort drop of OS page cache so the next dataset starts cold.
# `sudo -n` never prompts; if a password is required and not cached it just
# fails and we swallow the error. Safe no-op if sudo is unavailable.
sync || true
echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
