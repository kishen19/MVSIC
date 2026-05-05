#!/usr/bin/env bash
# Regenerate every plot under experiments/<stage>/results/_plots/ from the
# CSV/JSON/text results that are tracked in the repo.
#
# This is the one-shot entry point referenced by .gitignore: plot PDFs are
# untracked, this script reproduces them deterministically from the data
# files. Per-experiment plot scripts each have their own --datasets,
# --results, --out-dir flags; we just call them with the standard defaults.
#
# Usage:
#   experiments/plot_all.sh                  # everything
#   experiments/plot_all.sh --skip mvivf_ablation,query_compression
#   experiments/plot_all.sh --only latency,batch,multi_latency
#   experiments/plot_all.sh --dry-run        # print commands, do not run
#
# Notes:
# - Scripts that have no data simply print a warning and skip cleanly; this
#   wrapper keeps going (we use `|| true` per stage so one missing dataset
#   does not abort the rest).
# - mvivf_ablation needs (variant, stage) tuples; we discover them from the
#   results tree and dispatch through scripts/run_ablation.sh --task evaluate.
# - query_compression needs --dataset; we iterate over every per-dataset
#   results dir that has a CSV.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# All known per-experiment plot stages (used by --skip / --only).
ALL_STAGES=(
  builds
  latency
  batch
  multi_latency
  muvera_ablation
  svh_graph_ablation
  vamana_ablations
  optimizations
  quantization_compare
  overretrieve_singlevector_benchmark
  overretrieve_multivector_benchmark
  query_compression
  mvivf_ablation
  nq_study
)

SKIP=""
ONLY=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip)    SKIP="$2"; shift 2;;
    --only)    ONLY="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *)
      echo "[plot_all] unknown flag: $1" >&2
      exit 2
      ;;
  esac
done

is_in_csv() {
  # is_in_csv <needle> <comma_separated_haystack>
  local needle="$1" csv="$2"
  IFS=',' read -ra parts <<< "$csv"
  for p in "${parts[@]}"; do
    [[ "$(echo "$p" | xargs)" == "$needle" ]] && return 0
  done
  return 1
}

want_stage() {
  local s="$1"
  if [[ -n "$ONLY" ]]; then
    is_in_csv "$s" "$ONLY"
    return $?
  fi
  if [[ -n "$SKIP" ]] && is_in_csv "$s" "$SKIP"; then
    return 1
  fi
  return 0
}

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

run_or_warn() {
  # Forgive a single stage failure -- we want every other stage to still try.
  if ! run "$@"; then
    echo "[plot_all] WARN: command failed (continuing): $*" >&2
  fi
}

# -----------------------------------------------------------------------------
# 1) Build report (markdown + bar plots).
#    We read the *mirrored* build-stats tree under
#    experiments/builds/results/indexes/, which is what's tracked in the repo
#    and covers every dataset; the local results/indexes/ tree may only have
#    a subset of datasets actually built on this machine.
# -----------------------------------------------------------------------------
if want_stage builds; then
  echo "=== plot: builds ==="
  run_or_warn python3 "$REPO_ROOT/experiments/builds/scripts/build_report.py" \
    --indexes "$REPO_ROOT/experiments/builds/results/indexes"
fi

# -----------------------------------------------------------------------------
# 2) Latency / batch / multi_latency: thin wrappers around plot_stage.py.
# -----------------------------------------------------------------------------
for stage in latency batch multi_latency; do
  if want_stage "$stage"; then
    echo "=== plot: ${stage} ==="
    run_or_warn python3 "$REPO_ROOT/experiments/${stage}/scripts/plot.py"
    if [[ -f "$REPO_ROOT/experiments/${stage}/scripts/plot_breakdown.py" ]]; then
      run_or_warn python3 "$REPO_ROOT/experiments/${stage}/scripts/plot_breakdown.py"
    fi
  fi
done

# -----------------------------------------------------------------------------
# 3) Per-method ablations + benchmarks that have a single plot.py with
#    sensible default args.
# -----------------------------------------------------------------------------
for stage in muvera_ablation svh_graph_ablation vamana_ablations \
             optimizations quantization_compare \
             overretrieve_singlevector_benchmark \
             overretrieve_multivector_benchmark; do
  if want_stage "$stage"; then
    echo "=== plot: ${stage} ==="
    run_or_warn python3 "$REPO_ROOT/experiments/${stage}/scripts/plot.py"
  fi
done

# -----------------------------------------------------------------------------
# 4) query_compression: per-dataset, driven by whatever CSVs exist.
# -----------------------------------------------------------------------------
if want_stage query_compression; then
  echo "=== plot: query_compression ==="
  qc_root="$REPO_ROOT/experiments/query_compression/results"
  if [[ -d "$qc_root" ]]; then
    for ds_dir in "$qc_root"/*/; do
      ds="$(basename "$ds_dir")"
      [[ "$ds" == "_plots" ]] && continue
      # Need at least one of ball_carving.csv / wards.csv to plot.
      if [[ ! -f "$ds_dir/ball_carving.csv" && ! -f "$ds_dir/wards.csv" ]]; then
        continue
      fi
      run_or_warn python3 "$REPO_ROOT/experiments/query_compression/scripts/plot.py" \
        --dataset "$ds" --results-dir "$ds_dir"
    done
  fi
fi

# -----------------------------------------------------------------------------
# 5) mvivf_ablation: discover (dataset, variant, stage) triples from the
#    results tree and re-run scripts/run_ablation.sh --task evaluate, which
#    knows where to find {db,queries,gt} for each dataset and writes the
#    PDFs into experiments/mvivf_ablation/results/_plots/.
#
#    Layout (any of):
#      results/<ds>/stage<N>/<variant>/.../latency_*.csv      # regular stages
#      results/<ds>/stage<N>/.../latency_*.csv                # mixed stages 5-7
# -----------------------------------------------------------------------------
if want_stage mvivf_ablation; then
  echo "=== plot: mvivf_ablation ==="
  ablation_root="$REPO_ROOT/experiments/mvivf_ablation/results"
  if [[ -d "$ablation_root" ]]; then
    # Collect (ds, stage_n, variant) tuples; variant inferred from the path.
    declare -A SEEN
    while IFS= read -r csv; do
      rel="${csv#"$ablation_root"/}"
      ds="$(echo "$rel" | awk -F/ '{print $1}')"
      stage_dir="$(echo "$rel" | awk -F/ '{print $2}')"
      variant="$(echo "$rel" | awk -F/ '{print $3}')"
      # Stages 5/6/7 are mixed-method (variant directory is the second-level
      # build prefix, not one of mvivf/mvivf_spill/mvivf_flat). We dispatch
      # them as the "mvivf" variant; run_ablation.sh::evaluate_one detects
      # the mixed case via stage_is_mixed().
      case "$variant" in
        mvivf|mvivf_spill|mvivf_flat) ;;
        *) variant="mvivf" ;;
      esac
      [[ "$stage_dir" =~ ^stage([0-9]+)$ ]] || continue
      stage_n="${BASH_REMATCH[1]}"
      key="${ds}|${variant}|${stage_n}"
      [[ -n "${SEEN[$key]:-}" ]] && continue
      SEEN[$key]=1
      run_or_warn bash "$REPO_ROOT/experiments/mvivf_ablation/scripts/run_ablation.sh" \
        --task evaluate \
        --variant "$variant" \
        --dataset "$ds" \
        --stage "$stage_n"
    done < <(find "$ablation_root" -type f -name 'latency_*.csv' 2>/dev/null)
  fi
fi

# -----------------------------------------------------------------------------
# 6) nq_study: delegates to latency / batch / multi_latency wrappers; its
#    runner already knows the right --results / --out-dir overrides.
# -----------------------------------------------------------------------------
if want_stage nq_study; then
  echo "=== plot: nq_study ==="
  if [[ -x "$REPO_ROOT/experiments/nq_study/scripts/run_nq_study.sh" ]] \
       || [[ -f "$REPO_ROOT/experiments/nq_study/scripts/run_nq_study.sh" ]]; then
    run_or_warn bash "$REPO_ROOT/experiments/nq_study/scripts/run_nq_study.sh" --task plot
  fi
fi

echo "=== plot_all.sh done ==="
