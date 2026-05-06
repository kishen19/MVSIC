#!/usr/bin/env bash
# Run the single-thread per-query latency sweep.
#
# Usage (one dataset per invocation):
#   experiments/latency/scripts/run_latency.sh --dataset beir5         # all BEIR-5 shards (beir5.search.yaml)
#   experiments/latency/scripts/run_latency.sh --dataset beirbig       # quora/nq/hotpotqa (beirbig.search.yaml)
#   experiments/latency/scripts/run_latency.sh --dataset nq500k        # standalone (nq500k.search.yaml)
#   experiments/latency/scripts/run_latency.sh --dataset msmarco       # standalone (msmarco.search.yaml)
#   experiments/latency/scripts/run_latency.sh --dataset lotte         # standalone (lotte.search.yaml)
#   experiments/latency/scripts/run_latency.sh --dataset arguana       # one shard from beir5.search.yaml
#   experiments/latency/scripts/run_latency.sh --dataset nq --method mvivf
#   experiments/latency/scripts/run_latency.sh --dataset arguana --method mvivf_spill
#   experiments/latency/scripts/run_latency.sh --dataset vidore --method muvera
#   experiments/latency/scripts/run_latency.sh --dataset arguana --exclude mvivf
#   experiments/latency/scripts/run_latency.sh --dataset arguana --task plot   # only re-render plots
#   experiments/latency/scripts/run_latency.sh --dataset arguana --k 100         # search_config k=100
#
# ``--task <run|plot|all>`` (default ``all``):
#   run  = run benchmark_search only, no plotting.
#   plot = re-render plots from existing CSVs (no search).
#   all  = run, then plot (the previous default).
#
# ``--k <int>`` (default ``10``): pick which ``search_configs[].name == k=<int>``
# block to run. Currently only ``10`` and ``100`` are configured per method.
# Results land under ``<results>/<dataset>/<method>/<build>/<variant>/k=<int>/``
# (already isolated by directory) and plots under ``_plots/k=<int>/``.
#
# FastPlaid + IGP are opt-in: omit by default and for ``--method all``. Use
# ``--method fastplaid`` / ``--method igp`` or ``--with-fastplaid`` /
# ``--with-igp`` to include them (BEIR-5 + vidore only; see fastplaid_scope.sh).
#
# ``--exclude <name>[,<name>...]`` drops those indices[].name entries after the
# --method / --dataset filtering and FastPlaid scoping. It does not affect the
# FastPlaid opt-in path.

set -euo pipefail

# Force single-threaded BLAS / kernel pools for true per-query latency
# measurements. This affects all methods uniformly (MVSIC + FastPlaid).
# Must be set BEFORE python3 launches so PyTorch / NumPy / Rayon read them.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/experiments/latency/configs"
FILTER_PY="$REPO_ROOT/experiments/builds/scripts/filter_config.py"
# shellcheck disable=SC1091
source "$REPO_ROOT/experiments/builds/scripts/fastplaid_scope.sh"
cd "$REPO_ROOT"

DATASET=""
METHOD=""
EXCLUDE=""       # comma-separated indices[].name to drop after filtering
WITH_FASTPLAID=0
WITH_IGP=0
TASK="all"       # run | plot | all
K_VALUE=10       # search_config name `k=<K_VALUE>`; only 10 / 100 configured
EXTRA_ARGS=()

BEIR5_DATASETS=(nfcorpus scifact arguana scidocs fiqa)
BEIRBIG_DATASETS=(quora nq hotpotqa)
DATASET_ALIASES=(beir5 beirbig nq500k vidore msmarco lotte)

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
    --dataset) DATASET="$2"; shift 2;;
    --method)  METHOD="$2";  shift 2;;
    --exclude) EXCLUDE="$2"; shift 2;;
    --task)    TASK="$2";    shift 2;;
    --k)       K_VALUE="$2"; shift 2;;
    --with-fastplaid) WITH_FASTPLAID=1; shift;;
    --with-igp) WITH_IGP=1; shift;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

case "$TASK" in
  run|plot|all) ;;
  *) echo "Unknown --task '$TASK' (use run|plot|all)" >&2; exit 2;;
esac

case "$K_VALUE" in
  10|100) ;;
  *) echo "Unknown --k '$K_VALUE' (configured values: 10, 100)" >&2; exit 2;;
esac
SEARCH_NAME="k=$K_VALUE"

if [[ -z "$DATASET" ]]; then
  echo "Specify --dataset <name>. Aliases: ${DATASET_ALIASES[*]}; BEIR-5 shards: ${BEIR5_DATASETS[*]}; BEIR-big shards: ${BEIRBIG_DATASETS[*]}." >&2
  exit 2
fi

is_in() {
  local needle="$1"; shift
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

SRC_YAML=""
FILTER_DATASET=""
if is_in "$DATASET" "${DATASET_ALIASES[@]}"; then
  SRC_YAML="$CONFIGS_DIR/${DATASET}.search.yaml"
elif is_in "$DATASET" "${BEIR5_DATASETS[@]}"; then
  SRC_YAML="$CONFIGS_DIR/beir5.search.yaml"
  FILTER_DATASET="$DATASET"
elif is_in "$DATASET" "${BEIRBIG_DATASETS[@]}"; then
  SRC_YAML="$CONFIGS_DIR/beirbig.search.yaml"
  FILTER_DATASET="$DATASET"
else
  echo "Unknown --dataset '$DATASET'. Aliases: ${DATASET_ALIASES[*]}; BEIR-5 shards: ${BEIR5_DATASETS[*]}; BEIR-big shards: ${BEIRBIG_DATASETS[*]}." >&2
  exit 2
fi

[[ -f "$SRC_YAML" ]] || { echo "[error] config not found: $SRC_YAML" >&2; exit 2; }

if fastplaid_skip_fastplaid_method "$DATASET" "${METHOD:-}"; then
  echo "[warn] FastPlaid search only runs on the classic BEIR-5 shards (nfcorpus … fiqa); skipping." >&2
  exit 0
fi
if igp_skip_igp_method "$DATASET" "${METHOD:-}"; then
  echo "[warn] IGP search only runs on the classic BEIR-5 shards (nfcorpus … fiqa); skipping." >&2
  exit 0
fi

NEED_FILTER=0
[[ -n "$FILTER_DATASET" ]] && NEED_FILTER=1
[[ -n "${METHOD:-}" && "$METHOD" != "all" ]] && NEED_FILTER=1

CONFIG_PATH="$SRC_YAML"
if [[ "$NEED_FILTER" -eq 1 ]]; then
  TMP="$(mktemp "${TMPDIR:-/tmp}/main_latency.XXXXXX.yaml")"
  TEMP_YAMLS+=("$TMP")
  args=(python3 "$FILTER_PY" --in "$SRC_YAML" --out "$TMP")
  [[ -n "$FILTER_DATASET" ]] && args+=(--dataset "$FILTER_DATASET")
  [[ -n "${METHOD:-}" && "$METHOD" != "all" ]] && args+=(--method "$METHOD")
  if ! "${args[@]}"; then
    echo "[error] filter_config.py failed" >&2
    exit 2
  fi
  CONFIG_PATH="$TMP"
fi

eff_ds="${FILTER_DATASET:-}"
if [[ -z "$eff_ds" ]] && is_in "$DATASET" "${DATASET_ALIASES[@]}"; then
  eff_ds="$DATASET"
fi

if fastplaid_should_strip_after_filters "$eff_ds" "${METHOD:-}" "$WITH_FASTPLAID"; then
  strip_tmp="$(mktemp "${TMPDIR:-/tmp}/main_latency_stripfp.XXXXXX.yaml")"
  TEMP_YAMLS+=("$strip_tmp")
  if ! python3 "$FILTER_PY" --in "$CONFIG_PATH" --out "$strip_tmp" --strip-indices fastplaid; then
    echo "[error] filter_config.py --strip-indices failed" >&2
    exit 2
  fi
  CONFIG_PATH="$strip_tmp"
fi
if igp_should_strip_after_filters "$eff_ds" "${METHOD:-}" "$WITH_IGP"; then
  strip_tmp="$(mktemp "${TMPDIR:-/tmp}/main_latency_stripigp.XXXXXX.yaml")"
  TEMP_YAMLS+=("$strip_tmp")
  if ! python3 "$FILTER_PY" --in "$CONFIG_PATH" --out "$strip_tmp" --strip-indices igp; then
    echo "[error] filter_config.py --strip-indices igp failed" >&2
    exit 2
  fi
  CONFIG_PATH="$strip_tmp"
fi

if [[ -n "$EXCLUDE" ]]; then
  excl_tmp="$(mktemp "${TMPDIR:-/tmp}/main_latency_exclude.XXXXXX.yaml")"
  TEMP_YAMLS+=("$excl_tmp")
  if ! python3 "$FILTER_PY" --in "$CONFIG_PATH" --out "$excl_tmp" --strip-indices "$EXCLUDE"; then
    echo "[error] filter_config.py --strip-indices ($EXCLUDE) failed" >&2
    exit 2
  fi
  CONFIG_PATH="$excl_tmp"
fi

# Always pin to the requested search_config (k=10 / k=100). Done last so
# every prior filter (--method, --dataset, FastPlaid scope, --exclude) has
# already shrunk the YAML; if the requested k is missing in this dataset we
# end up with an empty `indices:` and benchmark_search will report no work.
search_tmp="$(mktemp "${TMPDIR:-/tmp}/main_latency_search.XXXXXX.yaml")"
TEMP_YAMLS+=("$search_tmp")
if ! python3 "$FILTER_PY" --in "$CONFIG_PATH" --out "$search_tmp" --search "$SEARCH_NAME"; then
  echo "[error] filter_config.py --search $SEARCH_NAME failed" >&2
  exit 2
fi
CONFIG_PATH="$search_tmp"

if [[ "$TASK" == "run" || "$TASK" == "all" ]]; then
  echo "=== Latency ($SEARCH_NAME): $CONFIG_PATH ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_search.py" \
      --config "$CONFIG_PATH" \
      --mode latency \
      "${EXTRA_ARGS[@]}"

  sync || true
  echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
fi

if [[ "$TASK" == "plot" || "$TASK" == "all" ]]; then
  echo "=== Plots ($SEARCH_NAME): $DATASET ==="
  python3 "$REPO_ROOT/experiments/latency/scripts/plot.py" \
      --datasets "$DATASET" --k "$K_VALUE" || true
fi
