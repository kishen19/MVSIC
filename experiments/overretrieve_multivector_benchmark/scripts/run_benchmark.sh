#!/usr/bin/env bash
# Run bench_chamfer_overretrieve on datasets listed in configs/datasets.yaml.
#
# Usage:
#   scripts/run_benchmark.sh
#   scripts/run_benchmark.sh --datasets arguana,nfcorpus
#   scripts/run_benchmark.sh --skip-build --skip-existing
#   scripts/run_benchmark.sh --data-root /path/to/beir
#   scripts/run_benchmark.sh -- -pq -rabitq
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"

CONFIG="$EXP_DIR/configs/datasets.yaml"
RESULTS_DIR="$EXP_DIR/results"
DATASETS_FILTER=""
DATA_ROOT_OVERRIDE="${DATA_ROOT:-}"
SKIP_BUILD=0
SKIP_EXISTING=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)        CONFIG="$2"; shift 2 ;;
    --datasets)      DATASETS_FILTER="$2"; shift 2 ;;
    --data-root)     DATA_ROOT_OVERRIDE="$2"; shift 2 ;;
    --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
    --skip-build)    SKIP_BUILD=1; shift ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    --)              shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help)
      sed -n '2,16p' "$0"; exit 0 ;;
    *)               EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi
mkdir -p "$RESULTS_DIR"

CONFIG_ROWS="$(
  python3 - "$CONFIG" "$DATA_ROOT_OVERRIDE" "$DATASETS_FILTER" "$REPO_ROOT" <<'PY'
import os, sys, yaml

cfg_path, data_root_override, ds_filter, repo_root = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

data_root = data_root_override or cfg.get("data_root") or ""
if not data_root:
    sys.exit("ERROR: data_root not set (config or DATA_ROOT/--data-root)")
if not os.path.isabs(data_root):
    data_root = os.path.join(repo_root, data_root)

defaults = cfg.get("defaults") or {}
default_metric   = str(defaults.get("metric", "IP"))
default_k        = int(defaults.get("k", 10))
default_pq       = bool(defaults.get("pq", False))
default_rabitq   = bool(defaults.get("rabitq", False))
default_pq_block = defaults.get("pq_block", "")
default_pq_k     = defaults.get("pq_k", "")
default_fs_block = defaults.get("fs_block", "")
default_rbits    = defaults.get("rbits", "")
default_num_query = defaults.get("num_query", 1000)
default_subsample_seed = defaults.get("query_subsample_seed", 42)
default_max_k_prime = defaults.get("max_k_prime", 1000)
default_k_growth = defaults.get("k_growth", 1.1)

wanted = {s.strip() for s in ds_filter.split(",") if s.strip()} if ds_filter else None

def _abs(p):
    return p if (not p or os.path.isabs(p)) else os.path.join(data_root, p)
def _str(v):
    return "" if v is None or v == "" else str(v)

for ds in (cfg.get("datasets") or []):
    name = ds["name"]
    if wanted is not None and name not in wanted:
        continue

    metric = str(ds.get("metric", default_metric))
    k = int(ds.get("k", default_k))

    points  = ds.get("points")  or os.path.join(data_root, name, f"{name}_points.pcs")
    queries = ds.get("queries") or os.path.join(data_root, name, f"{name}_queries.pcs")
    gt      = ds.get("gt")      or os.path.join(data_root, name, f"{name}_chamfer_neighbors.gt")

    pq       = bool(ds.get("pq", default_pq))
    rabitq   = bool(ds.get("rabitq", default_rabitq))
    pq_block = _str(ds.get("pq_block", default_pq_block))
    pq_k     = _str(ds.get("pq_k", default_pq_k))
    fs_block = _str(ds.get("fs_block", default_fs_block))
    rbits    = _str(ds.get("rbits", default_rbits))
    num_query = _str(ds.get("num_query", default_num_query))
    subsample_seed = _str(ds.get("query_subsample_seed", default_subsample_seed))
    max_k_prime = _str(ds.get("max_k_prime", default_max_k_prime))
    k_growth = _str(ds.get("k_growth", default_k_growth))
    extra    = " ".join(ds.get("extra_args") or [])

    print("\x1f".join([name, metric, str(k), _abs(points), _abs(queries), _abs(gt),
                       "1" if pq else "0", "1" if rabitq else "0",
                       pq_block, pq_k, fs_block, rbits, num_query, subsample_seed,
                       max_k_prime, k_growth, extra]))
PY
)"

if [[ -z "$CONFIG_ROWS" ]]; then
  echo "ERROR: no datasets matched (filter=$DATASETS_FILTER)" >&2
  exit 2
fi

BIN_LABEL="//microbenchmark/quantization:bench_chamfer_overretrieve"
BIN_PATH="$REPO_ROOT/bazel-bin/microbenchmark/quantization/bench_chamfer_overretrieve"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "=== bazel build $BIN_LABEL ==="
  ( cd "$REPO_ROOT" && bazel build "$BIN_LABEL" )
fi
if [[ ! -x "$BIN_PATH" ]]; then
  echo "ERROR: binary not found at $BIN_PATH (build failed?)" >&2
  exit 2
fi

SUMMARY="$RESULTS_DIR/summary.txt"
: > "$SUMMARY"
{
  echo "# Overretrieve Multi-Vector Benchmark — summary"
  echo "# config: $CONFIG"
  echo "# binary: $BIN_PATH"
  echo "# host:   $(hostname 2>/dev/null || echo unknown)"
  echo "# date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# git:    $(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown) ($(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown))"
  echo
} >> "$SUMMARY"

N_TOTAL="$(printf '%s\n' "$CONFIG_ROWS" | wc -l | tr -d ' ')"
i=0
while IFS=$'\x1f' read -r NAME METRIC K POINTS QUERIES GT PQ RABITQ PQ_BLOCK PQ_K FS_BLOCK RBITS NUM_QUERY SUBSAMPLE_SEED MAX_K_PRIME K_GROWTH DS_EXTRA; do
  i=$((i + 1))
  echo "=== [$i/$N_TOTAL] $NAME (metric=$METRIC, k=$K) ==="

  for f in "$POINTS" "$QUERIES"; do
    if [[ ! -f "$f" ]]; then
      echo "ERROR: missing input for '$NAME': $f" >&2
      echo "[$NAME] SKIPPED (missing $f)" >> "$SUMMARY"
      continue 2
    fi
  done
  if [[ ! -f "$GT" ]]; then
    echo "ERROR: missing GT for '$NAME': $GT" >&2
    echo "[$NAME] SKIPPED (missing GT $GT)" >> "$SUMMARY"
    continue
  fi

  OUT="$RESULTS_DIR/chamfer_${NAME}.txt"
  OK_MARKER="$RESULTS_DIR/chamfer_${NAME}.ok"
  if [[ "$SKIP_EXISTING" -eq 1 && -f "$OUT" && -f "$OK_MARKER" ]]; then
    echo "  -> skipping (results exist: $OUT)"
    echo "[$NAME] SKIPPED-EXISTING -> $OUT" >> "$SUMMARY"
    continue
  fi

  CMD=( "$BIN_PATH"
        -dist_func "$METRIC"
        -i "$POINTS"
        -q "$QUERIES"
        -gt "$GT"
        -k "$K" )
  [[ "$PQ" == "1"     ]] && CMD+=( -pq )
  [[ "$RABITQ" == "1" ]] && CMD+=( -rabitq )
  [[ -n "$PQ_BLOCK"   ]] && CMD+=( -pq_block "$PQ_BLOCK" )
  [[ -n "$PQ_K"       ]] && CMD+=( -pq_k "$PQ_K" )
  [[ -n "$FS_BLOCK"   ]] && CMD+=( -fs_block "$FS_BLOCK" )
  [[ -n "$RBITS"      ]] && CMD+=( -rbits "$RBITS" )
  [[ -n "$NUM_QUERY"  ]] && CMD+=( -num_query "$NUM_QUERY" )
  [[ -n "$SUBSAMPLE_SEED" ]] && CMD+=( -query_subsample_seed "$SUBSAMPLE_SEED" )
  [[ -n "$MAX_K_PRIME" ]] && CMD+=( -max_k_prime "$MAX_K_PRIME" )
  [[ -n "$K_GROWTH"    ]] && CMD+=( -k_growth "$K_GROWTH" )

  if [[ -n "$DS_EXTRA" ]]; then
    # shellcheck disable=SC2206
    DS_EXTRA_ARR=( $DS_EXTRA )
    CMD+=( "${DS_EXTRA_ARR[@]}" )
  fi
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    CMD+=( "${EXTRA_ARGS[@]}" )
  fi

  rm -f "$OK_MARKER"
  {
    echo "# command: ${CMD[*]}"
    echo "# host:    $(hostname 2>/dev/null || echo unknown)"
    echo "# date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
  } > "$OUT"

  if "${CMD[@]}" 2>&1 | tee -a "$OUT"; then
    : > "$OK_MARKER"
    echo "[chamfer_${NAME}] OK -> $OUT" >> "$SUMMARY"
  else
    rc=${PIPESTATUS[0]}
    echo "[chamfer_${NAME}] FAILED (rc=$rc) -> $OUT" >> "$SUMMARY"
    echo "ERROR: $NAME failed (rc=$rc); see $OUT" >&2
  fi
done <<< "$CONFIG_ROWS"

echo
echo "=== Done. Summary: $SUMMARY ==="
echo "Plot results:  python3 \"$EXP_DIR/scripts/plot.py\" --results-dir \"$RESULTS_DIR\""
