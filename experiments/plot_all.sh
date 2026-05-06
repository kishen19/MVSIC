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
#   experiments/plot_all.sh --only builds    # just the build report
#   experiments/plot_all.sh --suites all,beir   # which build-report suites to emit
#   experiments/plot_all.sh --suites beir       # BEIR-only build report
#   experiments/plot_all.sh --dry-run        # print commands, do not run
#   experiments/plot_all.sh --list-stages    # print the names accepted by --only / --skip
#
# Notes:
# - Scripts that have no data simply print a warning and skip cleanly; this
#   wrapper keeps going (we use `|| true` per stage so one missing dataset
#   does not abort the rest).
# - --suites only affects shared-across-datasets plots (currently the build
#   report); per-dataset PDFs always get one file per dataset, so suite
#   filtering would just be a no-op.
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
LIST_STAGES=0
# Default: emit the all-datasets, BEIR-only and LoTTE-only build reports.
# Use --suites to scope to just one (e.g. --suites beir).
SUITES="all,beir,lotte"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip)        SKIP="$2"; shift 2;;
    --only)        ONLY="$2"; shift 2;;
    --suites)      SUITES="$2"; shift 2;;
    --dry-run)     DRY_RUN=1; shift;;
    --list-stages) LIST_STAGES=1; shift;;
    -h|--help)
      sed -n '2,30p' "$0"
      echo
      echo "Available stages (for --only / --skip):"
      printf '  %s\n' "${ALL_STAGES[@]}"
      exit 0
      ;;
    *)
      echo "[plot_all] unknown flag: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$LIST_STAGES" -eq 1 ]]; then
  printf '%s\n' "${ALL_STAGES[@]}"
  exit 0
fi

is_in_csv() {
  # is_in_csv <needle> <comma_separated_haystack>
  local needle="$1" csv="$2"
  IFS=',' read -ra parts <<< "$csv"
  for p in "${parts[@]}"; do
    [[ "$(echo "$p" | xargs)" == "$needle" ]] && return 0
  done
  return 1
}

# Validate --only / --skip values against ALL_STAGES so a typo fails loudly
# instead of silently running everything (--skip) or nothing (--only).
validate_stage_csv() {
  local label="$1" csv="$2"
  [[ -z "$csv" ]] && return 0
  local bad=()
  IFS=',' read -ra parts <<< "$csv"
  for p in "${parts[@]}"; do
    p="$(echo "$p" | xargs)"
    [[ -z "$p" ]] && continue
    local found=0
    for s in "${ALL_STAGES[@]}"; do
      [[ "$s" == "$p" ]] && { found=1; break; }
    done
    [[ "$found" -eq 0 ]] && bad+=("$p")
  done
  if (( ${#bad[@]} > 0 )); then
    echo "[plot_all] $label: unknown stage(s): ${bad[*]}" >&2
    echo "[plot_all] known stages: ${ALL_STAGES[*]}" >&2
    exit 2
  fi
}
validate_stage_csv "--only" "$ONLY"
validate_stage_csv "--skip" "$SKIP"

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
#
#    The build report is the only "shared across datasets" plot in the repo
#    (single bar chart with one bar group per dataset), so the --suites flag
#    is consumed here. Each suite emits its own tagged outputs:
#      build_report.md         build_time.pdf         index_size.pdf
#      build_report_beir.md    build_time_beir.pdf    index_size_beir.pdf
#      build_report_vidore.md  build_time_vidore.pdf  index_size_vidore.pdf
#      build_report_lotte.md   build_time_lotte.pdf   index_size_lotte.pdf
# -----------------------------------------------------------------------------
if want_stage builds; then
  echo "=== plot: builds ==="
  IFS=',' read -ra SUITE_ARR <<< "$SUITES"
  for suite in "${SUITE_ARR[@]}"; do
    suite="$(echo "$suite" | xargs)"
    [[ -z "$suite" ]] && continue
    run_or_warn python3 "$REPO_ROOT/experiments/builds/scripts/build_report.py" \
      --indexes "$REPO_ROOT/experiments/builds/results/indexes" \
      --suite "$suite"
  done
fi

# -----------------------------------------------------------------------------
# 2) Latency / batch / multi_latency: thin wrappers around plot_stage.py.
#
#    Per-search-config (k=10 / k=100) results sit in sibling
#    ``<results>/<ds>/<method>/<build>/<variant>/k=<N>/`` directories. We
#    discover which `k=<N>` values actually exist on disk and loop the plot
#    wrapper once per value, so PDFs land under
#    ``<results>/_plots/k=<N>/<ds>_*.pdf`` and never mix Pareto fronts
#    across k. If no k=* dirs are present (legacy data) we fall back to a
#    single mixed render at ``<results>/_plots/``.
# -----------------------------------------------------------------------------
discover_k_values() {
  # Echo space-separated, sorted k values (just the integer) found under
  # any depth of the given results root. Empty output => no k=* dirs.
  local root="$1"
  [[ -d "$root" ]] || return 0
  find "$root" -type d -name 'k=*' 2>/dev/null \
    | sed 's|.*/k=||' \
    | grep -E '^[0-9]+$' \
    | sort -u -n
}

for stage in latency batch multi_latency; do
  if want_stage "$stage"; then
    echo "=== plot: ${stage} ==="
    stage_results="$REPO_ROOT/experiments/${stage}/results"
    mapfile -t K_VALUES < <(discover_k_values "$stage_results")
    if (( ${#K_VALUES[@]} == 0 )); then
      run_or_warn python3 "$REPO_ROOT/experiments/${stage}/scripts/plot.py"
    else
      for k in "${K_VALUES[@]}"; do
        echo "    --- k=$k ---"
        run_or_warn python3 "$REPO_ROOT/experiments/${stage}/scripts/plot.py" --k "$k"
      done
    fi
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
