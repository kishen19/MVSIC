#!/usr/bin/env bash
# Driver for flat brute-force query-compression τ sweeps (ball carving vs Ward).
#
# Usage:
#   scripts/run_query_compression.sh --dataset arguana --method all --task all
#   scripts/run_query_compression.sh --datasets arguana,nfcorpus --method ball --task sweep --tau_steps 24
#   scripts/run_query_compression.sh --dataset fiqa --task plot
#
# Forward sweep binary flags after a lone '--', or pass known extras via EXTRA_ARGS.
# Environment:
#   BAZEL_CMD   Default: bazel

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

BAZEL_CMD="${BAZEL_CMD:-bazel}"

DATASET=""
DATASETS=""
METHOD="all"
TASK="all"

EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --method) METHOD="$2"; shift 2;;
    --task) TASK="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done

if [[ -z "$DATASET" && -z "$DATASETS" ]]; then
  echo "Specify --dataset <name> or --datasets a,b,c" >&2
  exit 1
fi

run_one() {
  local name="$1"
  local base="$REPO_ROOT/data/beir/$name"
  local pts="$base/${name}_points.pcs"
  local q="$base/${name}_queries.pcs"
  local gt="$base/${name}_chamfer_neighbors.gt"
  local out_dir="$REPO_ROOT/experiments/query_compression/results/$name"
  mkdir -p "$out_dir/plots"

  if [[ "$TASK" == "plot" ]]; then
    python3 "$REPO_ROOT/experiments/query_compression/scripts/plot.py" \
      --dataset "$name" --results-dir "$out_dir" --out-dir "$out_dir/plots"
    return 0
  fi

  if [[ "$TASK" == "sweep" || "$TASK" == "all" ]]; then
    if [[ "$METHOD" == "ball" || "$METHOD" == "all" ]]; then
      echo "[run_query_compression] $name: ball carving -> $out_dir/ball_carving.csv"
      "$BAZEL_CMD" run //microbenchmark/query_compression:sweep_ball_carving -- \
        -i "$pts" -q "$q" -gt "$gt" -o "$out_dir/ball_carving.csv" "${EXTRA[@]}"
    fi
    if [[ "$METHOD" == "wards" || "$METHOD" == "all" ]]; then
      echo "[run_query_compression] $name: wards -> $out_dir/wards.csv"
      "$BAZEL_CMD" run //microbenchmark/query_compression:sweep_wards -- \
        -i "$pts" -q "$q" -gt "$gt" -o "$out_dir/wards.csv" "${EXTRA[@]}"
    fi
  fi

  if [[ "$TASK" == "plot" || "$TASK" == "all" ]]; then
    python3 "$REPO_ROOT/experiments/query_compression/scripts/plot.py" \
      --dataset "$name" --results-dir "$out_dir" --out-dir "$out_dir/plots"
  fi
}

case "$TASK" in
  sweep|plot|all) ;;
  *) echo "Unknown --task '$TASK' (use sweep|plot|all)" >&2; exit 1;;
esac

if [[ -n "$DATASET" ]]; then
  run_one "$DATASET"
else
  IFS=',' read -ra DS <<< "$DATASETS"
  for name in "${DS[@]}"; do
    name="$(echo "$name" | xargs)"
    [[ -z "$name" ]] && continue
    run_one "$name"
  done
fi
