#!/usr/bin/env bash
# Sourced by experiments/*/scripts/run_*.sh — FastPlaid scope for merged BEIR yaml.
# Indices stay in the checked-in YAML; runners strip ``fastplaid`` per shard when needed.

# Classic BEIR-5 corpora: FastPlaid is meaningful only on these merged-``beir`` shards.
FASTPLAID_BEIR_SHARDS=(nfcorpus scifact arguana scidocs fiqa)

# Names listed in ``experiments/builds/configs/beir.build.yaml`` / ``latency/configs/beir.search.yaml``.
# Keep in sync with BEIR5_DATASETS arrays in the run scripts.
BEIR_MERGED_NAMES=(nfcorpus scifact arguana scidocs fiqa quora nq hotpotqa nq500k)

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
  _fp_is_in "$ds" "${FASTPLAID_BEIR_SHARDS[@]}" && return 1
  _fp_is_in "$ds" "${BEIR_MERGED_NAMES[@]}" && return 0
  return 1
}

# Exit 0 => skip the whole run when user asked for ``--method fastplaid`` only.
fastplaid_skip_fastplaid_method() {
  local dataset="$1"
  local meth="${2:-}"
  [[ "$meth" == "fastplaid" ]] || return 1
  [[ "$dataset" == "vidore" ]] && return 1
  [[ "$dataset" == "msmarco" ]] && return 0
  _fp_is_in "$dataset" "${FASTPLAID_BEIR_SHARDS[@]}" && return 1
  _fp_is_in "$dataset" "${BEIR_MERGED_NAMES[@]}" && return 0
  return 1
}
