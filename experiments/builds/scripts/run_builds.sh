#!/usr/bin/env bash
# Build the indices for the main experiments.
#
# Usage (one dataset per invocation; use ``--method`` to pick one index or omit for all):
#   experiments/builds/scripts/run_builds.sh --dataset beir5            # all BEIR-5 shards (beir5.build.yaml)
#   experiments/builds/scripts/run_builds.sh --dataset beirbig          # quora/nq/hotpotqa (beirbig.build.yaml)
#   experiments/builds/scripts/run_builds.sh --dataset nq500k           # standalone (nq500k.build.yaml)
#   experiments/builds/scripts/run_builds.sh --dataset msmarco          # standalone (msmarco.build.yaml)
#   experiments/builds/scripts/run_builds.sh --dataset lotte            # standalone (lotte.build.yaml)
#   experiments/builds/scripts/run_builds.sh --dataset arguana          # one shard from beir5.build.yaml
#   experiments/builds/scripts/run_builds.sh --dataset nq --method mvivf
#   experiments/builds/scripts/run_builds.sh --dataset arguana --method mvivf_spill
#   experiments/builds/scripts/run_builds.sh --dataset vidore --method muvera
#   experiments/builds/scripts/run_builds.sh --dataset nfcorpus --method fastplaid
#   experiments/builds/scripts/run_builds.sh --dataset nfcorpus --method igp
#   experiments/builds/scripts/run_builds.sh --dataset arguana --exclude mvivf,muvera
#   experiments/builds/scripts/run_builds.sh --dataset beir5 --task plot   # rebuild stats/plots only, no build
#
# ``--task <build|plot|all>`` (default ``all``):
#   build = run benchmark_build only (no mirror, no report).
#   plot  = mirror results/indexes/<shard>/ into experiments/builds/results/
#           indexes/<shard>/ and refresh build_report.md / build_time.pdf /
#           index_size.pdf -- handy when you've added a build out-of-band
#           or just want to regenerate the report.
#   all   = build, then mirror, then report (the previous default).
#
# FastPlaid + IGP are opt-in for builds too: omitted by default / ``--method all``.
# Use ``--method fastplaid`` / ``--method igp`` or the corresponding
# ``--with-fastplaid`` / ``--with-igp`` flags (see fastplaid_scope.sh). Both are
# never built on beirbig / nq500k / msmarco / lotte even with opt-in flags.
#
# ``--exclude <name>[,<name>...]`` drops those indices[].name entries from the
# resolved config (after --method / --dataset filtering and FastPlaid scoping).
# It does not affect the FastPlaid opt-in path.
#
# Index binaries land at  results/indexes/<dataset>/<method>/<build_name>/index.bin .
# The configs reference that path; you may symlink results/indexes to scratch.
#
# After the build finishes successfully, this script ALSO:
#   1. Mirrors only the minimal metadata needed for build reporting:
#      results/indexes/<shard>/**/build_stats.json -> experiments/builds/
#      results/indexes/<shard>/**/build_stats.json (no binaries copied).
#   2. Runs experiments/builds/scripts/build_report.py over that mirrored view
#      to emit experiments/builds/results/{build_report.md, build_time.pdf,
#      index_size.pdf}.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/experiments/builds/configs"
FILTER_PY="$REPO_ROOT/experiments/builds/scripts/filter_config.py"
# shellcheck disable=SC1091
source "$REPO_ROOT/experiments/builds/scripts/fastplaid_scope.sh"
cd "$REPO_ROOT"

DATASET=""
METHOD=""        # empty | all | mvivf | mvivf_spill | muvera | vamana | svh_graph | fastplaid | igp
EXCLUDE=""       # comma-separated indices[].name to drop after filtering
WITH_FASTPLAID=0
WITH_IGP=0
# build = run benchmark_build only; plot = mirror+report only (no build);
# all   = build, then mirror, then report. Default = all so existing
# scripted runs keep behaving the same.
TASK="all"
EXTRA_ARGS=()

# Per-shard dataset names, grouped by the YAML they share. Keep in sync with
# BEIR5_NAMES / BEIRBIG_NAMES in fastplaid_scope.sh and with the {build,search}
# yaml filenames in experiments/{builds,latency}/configs.
BEIR5_DATASETS=(nfcorpus scifact arguana scidocs fiqa)
BEIRBIG_DATASETS=(quora nq hotpotqa)

# Top-level config aliases (each has a dedicated <ds>.build.yaml file and is
# accepted directly without a per-dataset name filter).
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
    --with-fastplaid) WITH_FASTPLAID=1; shift;;
    --with-igp) WITH_IGP=1; shift;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

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
  SRC_YAML="$CONFIGS_DIR/${DATASET}.build.yaml"
elif is_in "$DATASET" "${BEIR5_DATASETS[@]}"; then
  SRC_YAML="$CONFIGS_DIR/beir5.build.yaml"
  FILTER_DATASET="$DATASET"
elif is_in "$DATASET" "${BEIRBIG_DATASETS[@]}"; then
  SRC_YAML="$CONFIGS_DIR/beirbig.build.yaml"
  FILTER_DATASET="$DATASET"
else
  echo "Unknown --dataset '$DATASET'. Aliases: ${DATASET_ALIASES[*]}; BEIR-5 shards: ${BEIR5_DATASETS[*]}; BEIR-big shards: ${BEIRBIG_DATASETS[*]}." >&2
  exit 2
fi

if [[ ! -f "$SRC_YAML" ]]; then
  echo "[error] config not found: $SRC_YAML" >&2
  exit 2
fi

if fastplaid_skip_fastplaid_method "$DATASET" "${METHOD:-}"; then
  echo "[warn] FastPlaid builds only run on the classic BEIR-5 shards (nfcorpus … fiqa); skipping." >&2
  exit 0
fi
if igp_skip_igp_method "$DATASET" "${METHOD:-}"; then
  echo "[warn] IGP builds only run on the classic BEIR-5 shards (nfcorpus … fiqa); skipping." >&2
  exit 0
fi

NEED_FILTER=0
[[ -n "$FILTER_DATASET" ]] && NEED_FILTER=1
[[ -n "${METHOD:-}" && "$METHOD" != "all" ]] && NEED_FILTER=1

CONFIG_PATH="$SRC_YAML"
if [[ "$NEED_FILTER" -eq 1 ]]; then
  TMP="$(mktemp "${TMPDIR:-/tmp}/main_build.XXXXXX.yaml")"
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
  strip_tmp="$(mktemp "${TMPDIR:-/tmp}/main_build_stripfp.XXXXXX.yaml")"
  TEMP_YAMLS+=("$strip_tmp")
  if ! python3 "$FILTER_PY" --in "$CONFIG_PATH" --out "$strip_tmp" --strip-indices fastplaid; then
    echo "[error] filter_config.py --strip-indices failed" >&2
    exit 2
  fi
  CONFIG_PATH="$strip_tmp"
fi
if igp_should_strip_after_filters "$eff_ds" "${METHOD:-}" "$WITH_IGP"; then
  strip_tmp="$(mktemp "${TMPDIR:-/tmp}/main_build_stripigp.XXXXXX.yaml")"
  TEMP_YAMLS+=("$strip_tmp")
  if ! python3 "$FILTER_PY" --in "$CONFIG_PATH" --out "$strip_tmp" --strip-indices igp; then
    echo "[error] filter_config.py --strip-indices igp failed" >&2
    exit 2
  fi
  CONFIG_PATH="$strip_tmp"
fi

if [[ -n "$EXCLUDE" ]]; then
  excl_tmp="$(mktemp "${TMPDIR:-/tmp}/main_build_exclude.XXXXXX.yaml")"
  TEMP_YAMLS+=("$excl_tmp")
  if ! python3 "$FILTER_PY" --in "$CONFIG_PATH" --out "$excl_tmp" --strip-indices "$EXCLUDE"; then
    echo "[error] filter_config.py --strip-indices ($EXCLUDE) failed" >&2
    exit 2
  fi
  CONFIG_PATH="$excl_tmp"
fi

drop_caches_tail() {
  sync || true
  echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
}

# Determine the on-disk shards we just built so we can mirror them into
# experiments/builds/results/indexes/ and run the per-shard report. Mirrors
# the dispatch logic above: the alias 'beir5' covers BEIR5_DATASETS, etc.
shards_for_dataset() {
  local d="$1"
  if [[ "$d" == "beir5" ]]; then
    printf '%s\n' "${BEIR5_DATASETS[@]}"
    return
  fi
  if [[ "$d" == "beirbig" ]]; then
    printf '%s\n' "${BEIRBIG_DATASETS[@]}"
    return
  fi
  printf '%s\n' "$d"
}

case "$TASK" in
  build|plot|all) ;;
  *)
    echo "Unknown --task '$TASK' (use build|plot|all)." >&2
    exit 2
    ;;
esac

if [[ "$TASK" == "build" || "$TASK" == "all" ]]; then
  echo "=== Build: $CONFIG_PATH ==="
  python3 "$REPO_ROOT/benchmarks/benchmark_build.py" \
      --config "$CONFIG_PATH" \
      "${EXTRA_ARGS[@]}"
fi

# Mirror + report runs for both --task all and --task plot. The mirror
# step is a no-op if results/indexes/<shard>/ doesn't exist yet.
if [[ "$TASK" == "all" || "$TASK" == "plot" ]]; then
  # Mirror only build_stats.json files from
  # results/indexes/<shard>/**/build_stats.json into
  # experiments/builds/results/indexes/<shard>/**/build_stats.json.
  # This keeps the committed experiments tree lightweight (no index binaries)
  # while still providing everything build_report.py needs.
  INDEX_SRC_ROOT="$REPO_ROOT/results/indexes"
  INDEX_DST_ROOT="$REPO_ROOT/experiments/builds/results/indexes"
  shards_to_mirror=()
  if [[ -n "${FILTER_DATASET}" ]]; then
    shards_to_mirror=("${FILTER_DATASET}")
  else
    while IFS= read -r s; do shards_to_mirror+=("$s"); done < <(shards_for_dataset "$DATASET")
  fi
  mirror_build_stats_only() {
    local src_root="$1"
    local dst_root="$2"
    local shard="$3"
    python3 - "$src_root" "$dst_root" "$shard" << 'PY'
import pathlib
import shutil
import sys

src_root = pathlib.Path(sys.argv[1])
dst_root = pathlib.Path(sys.argv[2])
shard = sys.argv[3]

src = src_root / shard
dst = dst_root / shard
if not src.exists():
    print(f"[mirror] skip {shard}: {src} not found")
    raise SystemExit(0)

count = 0
for p in src.rglob("build_stats.json"):
    rel = p.relative_to(src)
    q = dst / rel
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, q)
    count += 1

print(f"[mirror] {shard}: copied {count} build_stats.json file(s)")
PY
  }

  mkdir -p "$INDEX_DST_ROOT"
  for shard in "${shards_to_mirror[@]}"; do
    mirror_build_stats_only "$INDEX_SRC_ROOT" "$INDEX_DST_ROOT" "$shard"
  done

  # Build report (markdown + bar plots) over the mirrored copy under
  # experiments/. Keeping the report next to the artifacts also means the file
  # is committed-friendly without bringing the binaries with it.
  echo "=== Build report (writes to experiments/builds/results/) ==="
  python3 "$REPO_ROOT/experiments/builds/scripts/build_report.py" \
      --indexes "$INDEX_DST_ROOT" \
      --out-dir "$REPO_ROOT/experiments/builds/results" \
    || echo "[warn] build_report.py failed; continuing"
fi

drop_caches_tail
