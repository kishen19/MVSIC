#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
LOG_DIR="sigmod_experiments/logs"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  "sigmod_experiments/configs/final_beir5_tq.yaml"
  "sigmod_experiments/configs/final_quora_tq.yaml"
  "sigmod_experiments/configs/final_nq_tq.yaml"
  "sigmod_experiments/configs/final_hotpotqa_tq.yaml"
  # "sigmod_experiments/configs/final_lotte_pooled_tq.yaml"
  "sigmod_experiments/configs/final_vidore_tq.yaml"
  "sigmod_experiments/configs/final_fastplaid_beir5.yaml"
)

for config in "${CONFIGS[@]}"; do
  config_name="$(basename "${config}" .yaml)"
  log_file="${LOG_DIR}/${config_name}.log"
  echo "============================================================"
  echo "Running benchmark with config: ${config}"
  echo "Log file: ${log_file}"
  echo "============================================================"
  python benchmarks/benchmark.py --config "${config}" 2>&1 | tee "${log_file}"
done

echo "All final benchmark runs completed."
