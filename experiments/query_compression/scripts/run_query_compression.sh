#!/usr/bin/env bash
# Driver for flat brute-force query-compression τ sweeps (ball carving vs Ward).
#
# Usage:
#   scripts/run_query_compression.sh --dataset arguana --method all --task all
#   scripts/run_query_compression.sh --datasets arguana,nfcorpus --method ball --task sweep --tau_steps 24
#   scripts/run_query_compression.sh --dataset fiqa --task plot
#   scripts/run_query_compression.sh --dataset fiqa -query_subsample 0  # disable subsample
#   scripts/run_query_compression.sh --dataset vidore --method wards --task all
#   scripts/run_query_compression.sh --datasets docvqa,chartqa --method all --task sweep
#
# Query subsampling: matches the latency / multi_latency / batch defaults
# plumbed through benchmarks/benchmark_search.py — the C++ sweep binaries
# default to ``-query_subsample 1000 -query_subsample_seed 42``. Pass
# ``-query_subsample 0`` to run on all queries (legacy behavior), or any
# other positive cap to override.
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

BEIR_DATASETS=(arguana fiqa nfcorpus quora scidocs scifact nq hotpotqa nq500k msmarco)
VIDORE_DATASETS=(docvqa infovqa arxivqa tabfquad chartqa shiftproject synth_ai synth_energy synth_gov synth_healthcare tatdqa)

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

is_in() {
  local needle="$1"; shift
  local x
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

resolve_dataset_base() {
  local name="$1"
  if [[ "$name" == "vidore" ]]; then
    echo "ALIAS_VIDORE"
    return 0
  fi
  if is_in "$name" "${VIDORE_DATASETS[@]}"; then
    echo "$REPO_ROOT/data/vidore/$name"
    return 0
  fi
  if is_in "$name" "${BEIR_DATASETS[@]}" || [[ -d "$REPO_ROOT/data/beir/$name" ]]; then
    echo "$REPO_ROOT/data/beir/$name"
    return 0
  fi
  echo ""
}

run_one() {
  local name="$1"
  local base
  base="$(resolve_dataset_base "$name")"
  if [[ -z "$base" || "$base" == "ALIAS_VIDORE" ]]; then
    echo "[run_query_compression] unknown dataset '$name'" >&2
    return 2
  fi
  local pts="$base/${name}_points.pcs"
  local q="$base/${name}_queries.pcs"
  local gt="$base/${name}_chamfer_neighbors.gt"
  local out_dir="$REPO_ROOT/experiments/query_compression/results/$name"
  # Canonical shared plot location, matching latency / batch / multi_latency /
  # quantization_compare / ... .  Per-dataset plots are disambiguated by
  # filename (recall_wards_<ds>.pdf, recall_ball_carving_<ds>.pdf).
  local plots_out_dir="$REPO_ROOT/experiments/query_compression/results/_plots"
  mkdir -p "$plots_out_dir"

  if [[ "$TASK" == "plot" ]]; then
    python3 "$REPO_ROOT/experiments/query_compression/scripts/plot.py" \
      --dataset "$name" --results-dir "$out_dir" --out-dir "$plots_out_dir"
    return 0
  fi

  if [[ "$TASK" == "sweep" || "$TASK" == "all" ]]; then
    if [[ ! -f "$pts" || ! -f "$q" || ! -f "$gt" ]]; then
      echo "[run_query_compression] missing inputs for '$name':" >&2
      [[ -f "$pts" ]] || echo "  - $pts" >&2
      [[ -f "$q" ]] || echo "  - $q" >&2
      [[ -f "$gt" ]] || echo "  - $gt" >&2
      return 2
    fi
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
      --dataset "$name" --results-dir "$out_dir" --out-dir "$plots_out_dir"
  fi
}

case "$TASK" in
  sweep|plot|all) ;;
  *) echo "Unknown --task '$TASK' (use sweep|plot|all)" >&2; exit 1;;
esac

if [[ -n "$DATASET" ]]; then
  if [[ "$DATASET" == "vidore" ]]; then
    for name in "${VIDORE_DATASETS[@]}"; do
      run_one "$name"
    done
  else
    run_one "$DATASET"
  fi
else
  IFS=',' read -ra DS <<< "$DATASETS"
  for name in "${DS[@]}"; do
    name="$(echo "$name" | xargs)"
    [[ -z "$name" ]] && continue
    if [[ "$name" == "vidore" ]]; then
      for vds in "${VIDORE_DATASETS[@]}"; do
        run_one "$vds"
      done
    else
      run_one "$name"
    fi
  done
fi
