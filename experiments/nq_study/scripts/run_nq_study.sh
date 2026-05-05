#!/usr/bin/env bash
# Run NQ-only study across build/latency/batch/multi-latency from one config set.
#
# Usage:
#   experiments/nq_study/scripts/run_nq_study.sh --task all
#   experiments/nq_study/scripts/run_nq_study.sh --task build --method mvivf
#   experiments/nq_study/scripts/run_nq_study.sh --task latency
#   experiments/nq_study/scripts/run_nq_study.sh --task plot
#
# Tasks:
#   build | latency | batch | multi_latency | plot | all
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FILTER_PY="$REPO_ROOT/experiments/builds/scripts/filter_config.py"
BUILD_CFG="$REPO_ROOT/experiments/nq_study/configs/nq.build.yaml"
SEARCH_CFG="$REPO_ROOT/experiments/nq_study/configs/nq.search.yaml"

TASK="all"
METHOD=""
EXCLUDE=""
EXTRA_ARGS=()

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
    --task) TASK="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --exclude) EXCLUDE="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

case "$TASK" in
  build|latency|batch|multi_latency|plot|all) ;;
  *)
    echo "Unknown --task '$TASK' (use build|latency|batch|multi_latency|plot|all)." >&2
    exit 2
    ;;
esac

prepare_cfg() {
  local src="$1"
  local dataset="$2"
  local method="$3"
  local exclude="$4"
  local cur="$src"

  if [[ -n "$dataset" || (-n "$method" && "$method" != "all") ]]; then
    local filt_tmp
    filt_tmp="$(mktemp "${TMPDIR:-/tmp}/nq_study_filt.XXXXXX.yaml")"
    TEMP_YAMLS+=("$filt_tmp")
    local args=(python3 "$FILTER_PY" --in "$cur" --out "$filt_tmp")
    [[ -n "$dataset" ]] && args+=(--dataset "$dataset")
    [[ -n "$method" && "$method" != "all" ]] && args+=(--method "$method")
    "${args[@]}"
    cur="$filt_tmp"
  fi

  if [[ -n "$exclude" ]]; then
    local ex_tmp
    ex_tmp="$(mktemp "${TMPDIR:-/tmp}/nq_study_exclude.XXXXXX.yaml")"
    TEMP_YAMLS+=("$ex_tmp")
    python3 "$FILTER_PY" --in "$cur" --out "$ex_tmp" --strip-indices "$exclude"
    cur="$ex_tmp"
  fi

  echo "$cur"
}

rewrite_stage_results_dir() {
  local src="$1"
  local stage="$2"
  local out
  out="$(mktemp "${TMPDIR:-/tmp}/nq_study_stage.XXXXXX.yaml")"
  TEMP_YAMLS+=("$out")
  python3 - "$src" "$out" "$stage" <<'PY'
import sys
import yaml

src, dst, stage = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
for d in cfg.get("datasets") or []:
    if not isinstance(d, dict):
        continue
    rd = d.get("results_dir")
    if isinstance(rd, str):
        d["results_dir"] = rd.replace("/latency/", f"/{stage}/")
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
PY
  echo "$out"
}

run_search_stage() {
  local mode="$1"
  local filtered_cfg="$2"
  local stage_cfg
  stage_cfg="$(rewrite_stage_results_dir "$filtered_cfg" "$mode")"
  echo "=== NQ study: $mode ($stage_cfg) ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_search.py" \
    --config "$stage_cfg" \
    --mode "$mode" \
    "${EXTRA_ARGS[@]}"
}

cd "$REPO_ROOT"

build_cfg="$(prepare_cfg "$BUILD_CFG" "nq" "$METHOD" "$EXCLUDE")"
search_cfg="$(prepare_cfg "$SEARCH_CFG" "nq" "$METHOD" "$EXCLUDE")"

if [[ "$TASK" == "build" || "$TASK" == "all" ]]; then
  echo "=== NQ study: build ($build_cfg) ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_build.py" \
    --config "$build_cfg" \
    "${EXTRA_ARGS[@]}"
fi

if [[ "$TASK" == "latency" || "$TASK" == "all" ]]; then
  run_search_stage "latency" "$search_cfg"
fi
if [[ "$TASK" == "batch" || "$TASK" == "all" ]]; then
  run_search_stage "batch" "$search_cfg"
fi
if [[ "$TASK" == "multi_latency" || "$TASK" == "all" ]]; then
  run_search_stage "multi_latency" "$search_cfg"
fi

if [[ "$TASK" == "plot" || "$TASK" == "all" ]]; then
  echo "=== NQ study: plots ==="
  python3 "$REPO_ROOT/experiments/latency/scripts/plot.py" \
    --datasets nq \
    --results "$REPO_ROOT/experiments/nq_study/results/latency" \
    --out-dir "$REPO_ROOT/experiments/nq_study/results/latency/_plots" || true
  python3 "$REPO_ROOT/experiments/batch/scripts/plot.py" \
    --datasets nq \
    --results "$REPO_ROOT/experiments/nq_study/results/batch" \
    --out-dir "$REPO_ROOT/experiments/nq_study/results/batch/_plots" || true
  python3 "$REPO_ROOT/experiments/multi_latency/scripts/plot.py" \
    --datasets nq \
    --results "$REPO_ROOT/experiments/nq_study/results/multi_latency" \
    --out-dir "$REPO_ROOT/experiments/nq_study/results/multi_latency/_plots" || true
fi
