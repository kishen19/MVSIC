#!/usr/bin/env bash
# Run the measure_stretch_tq quality microbenchmark on every dataset listed
# in configs/datasets.yaml and write per-dataset output to results/.
#
# Usage:
#   scripts/run_benchmark.sh                                    # all datasets, default config
#   scripts/run_benchmark.sh --datasets glove-100-angular       # subset
#   scripts/run_benchmark.sh --data-root /some/path             # override data_root
#   scripts/run_benchmark.sh --skip-build                       # don't re-run bazel build
#   scripts/run_benchmark.sh --config <yaml>                    # alternate config file
#   scripts/run_benchmark.sh -- -pq_method TQ4 -num_query 500   # forward extra binary flags
#
# Reproducibility notes:
#   * Repo root is derived from this script's location, so the script works
#     after a fresh clone with no environment setup beyond bazel + python3-yaml.
#   * The benchmark binary is rebuilt with bazel (-c opt, -march=native by
#     default; see .bazelrc) before any dataset runs, unless --skip-build.
#   * DATA_ROOT can be overridden at runtime; otherwise the YAML value is used.
#   * Per-dataset stdout is captured verbatim into results/<dataset>.txt and a
#     combined summary is written to results/summary.txt. Re-running overwrites.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"

CONFIG="$EXP_DIR/configs/datasets.yaml"
RESULTS_DIR="$EXP_DIR/results"
DATASETS_FILTER=""
DATA_ROOT_OVERRIDE="${DATA_ROOT:-}"
SKIP_BUILD=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG="$2"; shift 2 ;;
    --datasets)   DATASETS_FILTER="$2"; shift 2 ;;
    --data-root)  DATA_ROOT_OVERRIDE="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --)           shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help)
      sed -n '2,21p' "$0"; exit 0 ;;
    *)            EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Parse YAML via python3+PyYAML and emit shell-friendly tab-separated lines:
#   name<TAB>metric<TAB>k<TAB>points<TAB>queries<TAB>gt<TAB>pq_method<TAB>
#   rabitq_bits<TAB>num_query<TAB>max_k_prime<TAB>k_growth<TAB>extra_args
# Unset optional fields are emitted as the empty string.
# ---------------------------------------------------------------------------
CONFIG_ROWS="$(
  python3 - "$CONFIG" "$DATA_ROOT_OVERRIDE" "$DATASETS_FILTER" <<'PY'
import os, sys, yaml

cfg_path, data_root_override, ds_filter = sys.argv[1], sys.argv[2], sys.argv[3]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f) or {}

data_root = data_root_override or cfg.get("data_root") or ""
if not data_root:
    sys.exit("ERROR: data_root not set (config or DATA_ROOT/--data-root)")

defaults = cfg.get("defaults") or {}
default_metric      = defaults.get("metric", "IP")
default_k           = int(defaults.get("k", 10))
default_pq_method   = defaults.get("pq_method", "")
default_rabitq_bits = defaults.get("rabitq_bits", "")
default_num_query   = defaults.get("num_query", "")
default_max_k_prime = defaults.get("max_k_prime", "")
default_k_growth    = defaults.get("k_growth", "")

wanted = {s.strip() for s in ds_filter.split(",") if s.strip()} if ds_filter else None

def _abs(p):
    return p if (not p or os.path.isabs(p)) else os.path.join(data_root, p)

def _str(v):
    return "" if v is None or v == "" else str(v)

for ds in (cfg.get("datasets") or []):
    name = ds["name"]
    if wanted is not None and name not in wanted:
        continue
    metric = ds.get("metric", default_metric)
    k      = int(ds.get("k", default_k))
    points  = ds.get("points")  or os.path.join(data_root, name, f"{name}_base.fbin")
    queries = ds.get("queries") or os.path.join(data_root, name, f"{name}_query.fbin")
    # GT is optional. If neither per-dataset nor convention file exists, the
    # binary will brute-force and cache it to /tmp.
    gt_explicit = ds.get("gt")
    if gt_explicit:
        gt = _abs(gt_explicit)
    else:
        conv_gt = os.path.join(data_root, name, f"{name}_groundtruth")
        gt = conv_gt if os.path.exists(conv_gt) else ""

    pq_method   = _str(ds.get("pq_method",   default_pq_method))
    rabitq_bits = _str(ds.get("rabitq_bits", default_rabitq_bits))
    num_query   = _str(ds.get("num_query",   default_num_query))
    max_k_prime = _str(ds.get("max_k_prime", default_max_k_prime))
    k_growth    = _str(ds.get("k_growth",    default_k_growth))
    extra       = " ".join(ds.get("extra_args") or [])

    print("\t".join([name, metric, str(k), _abs(points), _abs(queries), gt,
                     pq_method, rabitq_bits, num_query, max_k_prime, k_growth,
                     extra]))
PY
)"

if [[ -z "$CONFIG_ROWS" ]]; then
  echo "ERROR: no datasets matched (filter=$DATASETS_FILTER)" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Build the binary (deterministic flags come from .bazelrc).
# ---------------------------------------------------------------------------
BIN_LABEL="//microbenchmark/quantization:measure_stretch_tq"
BIN_PATH="$REPO_ROOT/bazel-bin/microbenchmark/quantization/measure_stretch_tq"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "=== bazel build $BIN_LABEL ==="
  ( cd "$REPO_ROOT" && bazel build "$BIN_LABEL" )
fi
if [[ ! -x "$BIN_PATH" ]]; then
  echo "ERROR: binary not found at $BIN_PATH (build failed?)" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Run per dataset.
# ---------------------------------------------------------------------------
SUMMARY="$RESULTS_DIR/summary.txt"
: > "$SUMMARY"

# Header for the summary log (helps verify reproducibility across machines).
{
  echo "# Measure Stretch Benchmark — summary"
  echo "# config: $CONFIG"
  echo "# binary: $BIN_PATH"
  echo "# host:   $(hostname 2>/dev/null || echo unknown)"
  echo "# date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# git:    $(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown) ($(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown))"
  echo
} >> "$SUMMARY"

# Count datasets up-front for "[i/N]" progress.
N_TOTAL="$(printf '%s\n' "$CONFIG_ROWS" | wc -l | tr -d ' ')"
i=0
while IFS=$'\t' read -r NAME METRIC K POINTS QUERIES GT PQ_METHOD RABITQ_BITS NUM_QUERY MAX_KP K_GROWTH DS_EXTRA; do
  i=$((i + 1))
  echo "=== [$i/$N_TOTAL] $NAME (metric=$METRIC, k=$K) ==="

  # Required inputs: points + queries. (GT is optional; binary auto-caches.)
  for f in "$POINTS" "$QUERIES"; do
    if [[ ! -f "$f" ]]; then
      echo "ERROR: missing input for '$NAME': $f" >&2
      echo "[$NAME] SKIPPED (missing $f)" >> "$SUMMARY"
      continue 2
    fi
  done
  # If a GT path was specified but doesn't exist, treat as a config error.
  if [[ -n "$GT" && ! -f "$GT" ]]; then
    echo "ERROR: GT path set for '$NAME' but file is missing: $GT" >&2
    echo "[$NAME] SKIPPED (missing GT $GT)" >> "$SUMMARY"
    continue
  fi

  OUT="$RESULTS_DIR/$NAME.txt"
  CMD=( "$BIN_PATH"
        -dist_func "$METRIC"
        -i "$POINTS"
        -q "$QUERIES"
        -k "$K" )
  [[ -n "$GT"          ]] && CMD+=( -gt          "$GT" )
  [[ -n "$PQ_METHOD"   ]] && CMD+=( -pq_method   "$PQ_METHOD" )
  [[ -n "$RABITQ_BITS" ]] && CMD+=( -rabitq_bits "$RABITQ_BITS" )
  [[ -n "$NUM_QUERY"   ]] && CMD+=( -num_query   "$NUM_QUERY" )
  [[ -n "$MAX_KP"      ]] && CMD+=( -max_k_prime "$MAX_KP" )
  [[ -n "$K_GROWTH"    ]] && CMD+=( -k_growth    "$K_GROWTH" )

  # Per-dataset extra_args (from YAML), then global EXTRA_ARGS (from CLI).
  if [[ -n "$DS_EXTRA" ]]; then
    # shellcheck disable=SC2206
    DS_EXTRA_ARR=( $DS_EXTRA )
    CMD+=( "${DS_EXTRA_ARR[@]}" )
  fi
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    CMD+=( "${EXTRA_ARGS[@]}" )
  fi

  {
    echo "# command: ${CMD[*]}"
    echo "# host:    $(hostname 2>/dev/null || echo unknown)"
    echo "# date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
  } > "$OUT"

  # Run, mirroring stdout to console and to the per-dataset file.
  if "${CMD[@]}" 2>&1 | tee -a "$OUT"; then
    echo "[$NAME] OK -> $OUT" >> "$SUMMARY"
  else
    rc=${PIPESTATUS[0]}
    echo "[$NAME] FAILED (rc=$rc) -> $OUT" >> "$SUMMARY"
    echo "ERROR: $NAME failed (rc=$rc); see $OUT" >&2
  fi
done <<< "$CONFIG_ROWS"

echo
echo "=== Done. Summary: $SUMMARY ==="
