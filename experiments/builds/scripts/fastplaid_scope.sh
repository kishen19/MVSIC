#!/usr/bin/env bash
# Sourced by experiments/*/scripts/run_*.sh — FastPlaid scope across the
# split BEIR configs (beir5 / beirbig / nq500k) plus the standalone shards.
# Keep these arrays in sync with the {build,search} YAML filenames in
# experiments/builds/configs and experiments/latency/configs.

# Classic BEIR-5 corpora: FastPlaid is meaningful only on these merged-``beir5`` shards.
FASTPLAID_BEIR_SHARDS=(nfcorpus scifact arguana scidocs fiqa)

# Datasets that share experiments/builds/configs/beir5.{build,search}.yaml.
BEIR5_NAMES=(nfcorpus scifact arguana scidocs fiqa)

# Datasets that share experiments/builds/configs/beirbig.{build,search}.yaml.
# FastPlaid is intentionally never built/run on these (too large).
BEIRBIG_NAMES=(quora nq hotpotqa)

_fp_is_in() {
  local needle="$1"; shift
  local x
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

# Exit 0 => remove ``fastplaid`` from ``indices`` for this dataset name.
fastplaid_should_strip_indices() {
  local ds="$1"
  [[ "$ds" == "msmarco" ]] && return 0
  [[ "$ds" == "nq500k" ]] && return 0
  _fp_is_in "$ds" "${FASTPLAID_BEIR_SHARDS[@]}" && return 1
  _fp_is_in "$ds" "${BEIRBIG_NAMES[@]}" && return 0
  return 1
}

# Exit 0 => strip ``fastplaid`` from the filtered config before benchmark_search / benchmark_build.
#
# FastPlaid is **opt-in** for allowed datasets: default ``--dataset arguana`` (no ``--method``)
# and ``--method all`` both strip FastPlaid. Enable via ``--method fastplaid`` or
# ``--with-fastplaid``. Dataset-level bans from ``fastplaid_should_strip_indices`` (e.g. nq,
# msmarco, nq500k) **always** strip; ``--with-fastplaid`` cannot override that.
#
# Args: effective_dataset  method  with_fastplaid_flag (0 or 1)
fastplaid_should_strip_after_filters() {
  local eff_ds="$1"
  local meth="${2:-}"
  local with_fp="${3:-0}"

  if [[ -n "$eff_ds" ]] && fastplaid_should_strip_indices "$eff_ds"; then
    return 0
  fi

  [[ "$meth" == "fastplaid" ]] && return 1

  [[ "$with_fp" == "1" ]] && return 1

  return 0
}

# Exit 0 => skip the whole run when user asked for ``--method fastplaid`` only.
fastplaid_skip_fastplaid_method() {
  local dataset="$1"
  local meth="${2:-}"
  [[ "$meth" == "fastplaid" ]] || return 1
  [[ "$dataset" == "vidore" ]] && return 1
  [[ "$dataset" == "msmarco" ]] && return 0
  [[ "$dataset" == "nq500k" ]] && return 0
  _fp_is_in "$dataset" "${FASTPLAID_BEIR_SHARDS[@]}" && return 1
  _fp_is_in "$dataset" "${BEIRBIG_NAMES[@]}" && return 0
  # Aliases that resolve to a top-level YAML (beir5/beirbig).
  [[ "$dataset" == "beir5" ]] && return 1
  [[ "$dataset" == "beirbig" ]] && return 0
  return 1
}
