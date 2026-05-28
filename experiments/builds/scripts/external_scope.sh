#!/usr/bin/env bash
# Sourced by experiments/*/scripts/run_*.sh — external baseline scope across the
# split BEIR configs (beir5 / beirbig / nq500k) plus standalone shards.
# Baselines: fastplaid, igp, gem, hnswlib (BEIR-5 only unless noted).
# Keep these arrays in sync with the {build,search} YAML filenames in
# experiments/builds/configs and experiments/latency/configs.

# Classic BEIR-5 corpora: external baselines are meaningful only on these
# merged ``beir5`` shards.
EXTERNAL_BEIR_SHARDS=(nfcorpus scifact arguana scidocs fiqa)

# Datasets that share experiments/builds/configs/beir5.{build,search}.yaml.
BEIR5_NAMES=(nfcorpus scifact arguana scidocs fiqa)

# Datasets that share experiments/builds/configs/beirbig.{build,search}.yaml.
# External baselines are intentionally never built/run on these (too large).
BEIRBIG_NAMES=(quora nq hotpotqa)

_external_is_in() {
  local needle="$1"; shift
  local x
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

# Exit 0 => remove this external baseline from ``indices`` for this dataset name.
_external_should_strip_indices() {
  local ds="$1"
  [[ "$ds" == "msmarco" ]] && return 0
  [[ "$ds" == "nq500k" ]] && return 0
  [[ "$ds" == "lotte" ]] && return 0
  _external_is_in "$ds" "${EXTERNAL_BEIR_SHARDS[@]}" && return 1
  _external_is_in "$ds" "${BEIRBIG_NAMES[@]}" && return 0
  return 1
}

# Exit 0 => remove ``fastplaid`` from ``indices`` for this dataset name.
fastplaid_should_strip_indices() {
  _external_should_strip_indices "$1"
}

# Exit 0 => remove ``igp`` from ``indices`` for this dataset name.
igp_should_strip_indices() {
  _external_should_strip_indices "$1"
}

# Exit 0 => remove ``gem`` from ``indices`` for this dataset name.
gem_should_strip_indices() {
  _external_should_strip_indices "$1"
}

# Exit 0 => remove ``hnswlib`` from ``indices`` for this dataset name.
hnswlib_should_strip_indices() {
  _external_should_strip_indices "$1"
}

# Args: baseline_name  effective_dataset  method  with_flag (0 or 1)
_external_should_strip_after_filters() {
  local baseline="$1"
  local eff_ds="$2"
  local meth="${3:-}"
  local with_baseline="${4:-0}"

  if [[ -n "$eff_ds" ]] && _external_should_strip_indices "$eff_ds"; then
    return 0
  fi

  [[ "$meth" == "$baseline" ]] && return 1
  [[ "$with_baseline" == "1" ]] && return 1
  return 0
}

# Exit 0 => strip ``fastplaid`` from the filtered config before benchmark_search / benchmark_build.
#
# FastPlaid is **opt-in** for allowed datasets: default ``--dataset arguana`` (no ``--method``)
# and ``--method all`` both strip FastPlaid. Enable via ``--method fastplaid`` or
# ``--with-fastplaid``. Dataset-level bans **always** strip; ``--with-fastplaid`` cannot override.
#
# Args: effective_dataset  method  with_fastplaid_flag (0 or 1)
fastplaid_should_strip_after_filters() {
  _external_should_strip_after_filters "fastplaid" "$1" "${2:-}" "${3:-0}"
}

# Exit 0 => strip ``igp`` from the filtered config before benchmark_search / benchmark_build.
igp_should_strip_after_filters() {
  _external_should_strip_after_filters "igp" "$1" "${2:-}" "${3:-0}"
}

# Exit 0 => strip ``gem`` from the filtered config before benchmark_search / benchmark_build.
gem_should_strip_after_filters() {
  _external_should_strip_after_filters "gem" "$1" "${2:-}" "${3:-0}"
}

# Exit 0 => strip ``hnswlib`` from the filtered config before benchmark_search / benchmark_build.
hnswlib_should_strip_after_filters() {
  _external_should_strip_after_filters "hnswlib" "$1" "${2:-}" "${3:-0}"
}

# Exit 0 => skip the whole run when user asked for this external baseline only.
_external_skip_method() {
  local baseline="$1"
  local dataset="$2"
  local meth="${3:-}"
  [[ "$meth" == "$baseline" ]] || return 1
  [[ "$dataset" == "vidore" ]] && return 1
  [[ "$dataset" == "msmarco" ]] && return 0
  [[ "$dataset" == "nq500k" ]] && return 0
  [[ "$dataset" == "lotte" ]] && return 0
  _external_is_in "$dataset" "${EXTERNAL_BEIR_SHARDS[@]}" && return 1
  _external_is_in "$dataset" "${BEIRBIG_NAMES[@]}" && return 0
  [[ "$dataset" == "beir5" ]] && return 1
  [[ "$dataset" == "beirbig" ]] && return 0
  return 1
}

fastplaid_skip_fastplaid_method() {
  _external_skip_method "fastplaid" "$1" "${2:-}"
}

igp_skip_igp_method() {
  _external_skip_method "igp" "$1" "${2:-}"
}

gem_skip_gem_method() {
  _external_skip_method "gem" "$1" "${2:-}"
}

hnswlib_skip_hnswlib_method() {
  _external_skip_method "hnswlib" "$1" "${2:-}"
}
