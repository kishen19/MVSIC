#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." &>/dev/null && pwd)"

for dir in "$REPO_ROOT"/data/beir/*/; do
    dataset="$(basename "$dir")"
    echo "=== Running benchmark on $dataset ==="
    "$SCRIPT_DIR/run_benchmark.sh" --datasets "$dataset" "$@"
done
