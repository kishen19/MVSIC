#!/usr/bin/env bash
# Move listed JSON files from ./results/ into experiments/builds/results/,
# preserving the path structure under results/.
set -euo pipefail

cd "$(dirname "$0")/../../.."  # repo root (MVSIC)

DEST_ROOT="experiments/builds/results"

FILES=(
  "results/indexes/fiqa/vamana/vamana_R64_L128_a1.0/index_params.json"
  "results/indexes/fiqa/vamana/vamana_R64_L128_a1.0/build_stats.json"
  "results/indexes/fiqa/muvera/muvera_10240_R200_L600_a1.0/index_params.json"
  "results/indexes/fiqa/muvera/muvera_10240_R200_L600_a1.0/build_stats.json"
  "results/indexes/fiqa/svh_graph/svh_graph_R64_L128_a1.0/index_params.json"
  "results/indexes/fiqa/svh_graph/svh_graph_R64_L128_a1.0/build_stats.json"
  "results/indexes/fiqa/mvivf/mvivf_k0_l100_d0_nit5_s0_mpcc100/index_params.json"
  "results/indexes/fiqa/mvivf/mvivf_k0_l100_d0_nit5_s0_mpcc100/build_stats.json"
  "results/indexes/nfcorpus/vamana/vamana_R64_L128_a1.0/index_params.json"
  "results/indexes/nfcorpus/vamana/vamana_R64_L128_a1.0/build_stats.json"
  "results/indexes/nfcorpus/muvera/muvera_10240_R200_L600_a1.0/index_params.json"
  "results/indexes/nfcorpus/muvera/muvera_10240_R200_L600_a1.0/build_stats.json"
  "results/indexes/nfcorpus/svh_graph/svh_graph_R64_L128_a1.0/index_params.json"
  "results/indexes/nfcorpus/svh_graph/svh_graph_R64_L128_a1.0/build_stats.json"
  "results/indexes/nfcorpus/mvivf/mvivf_k0_l100_d0_nit5_s0_mpcc100/index_params.json"
  "results/indexes/nfcorpus/mvivf/mvivf_k0_l100_d0_nit5_s0_mpcc100/build_stats.json"
  "results/indexes/arguana/vamana/vamana_R64_L128_a1.0/index_params.json"
  "results/indexes/arguana/vamana/vamana_R64_L128_a1.0/build_stats.json"
  "results/indexes/arguana/fastplaid/fastplaid_nbits4/merged_residuals.manifest.json"
  "results/indexes/arguana/fastplaid/fastplaid_nbits4/doclens.0.json"
  "results/indexes/arguana/fastplaid/fastplaid_nbits4/merged_codes.manifest.json"
  "results/indexes/arguana/fastplaid/fastplaid_nbits4/0.metadata.json"
  "results/indexes/arguana/fastplaid/fastplaid_nbits4/metadata.json"
  "results/indexes/arguana/fastplaid/fastplaid_nbits4/plan.json"
  "results/indexes/arguana/fastplaid/fastplaid_nbits4/build_stats.json"
  "results/indexes/arguana/muvera/muvera_10240_R200_L600_a1.0/index_params.json"
  "results/indexes/arguana/muvera/muvera_10240_R200_L600_a1.0/build_stats.json"
  "results/indexes/arguana/svh_graph/svh_graph_R64_L128_a1.0/index_params.json"
  "results/indexes/arguana/svh_graph/svh_graph_R64_L128_a1.0/build_stats.json"
  "results/indexes/arguana/mvivf/mvivf_k0_l100_d0_nit5_s0_mpcc100/index_params.json"
  "results/indexes/arguana/mvivf/mvivf_k0_l100_d0_nit5_s0_mpcc100/build_stats.json"
  "results/indexes/scifact/vamana/vamana_R64_L128_a1.0/index_params.json"
  "results/indexes/scifact/vamana/vamana_R64_L128_a1.0/build_stats.json"
  "results/indexes/scifact/muvera/muvera_10240_R200_L600_a1.0/index_params.json"
  "results/indexes/scifact/muvera/muvera_10240_R200_L600_a1.0/build_stats.json"
  "results/indexes/scifact/svh_graph/svh_graph_R64_L128_a1.0/index_params.json"
  "results/indexes/scifact/svh_graph/svh_graph_R64_L128_a1.0/build_stats.json"
  "results/indexes/scifact/mvivf/mvivf_k0_l100_d0_nit5_s0_mpcc100/index_params.json"
  "results/indexes/scifact/mvivf/mvivf_k0_l100_d0_nit5_s0_mpcc100/build_stats.json"
  "results/indexes/scidocs/vamana/vamana_R64_L128_a1.0/index_params.json"
  "results/indexes/scidocs/vamana/vamana_R64_L128_a1.0/build_stats.json"
  "results/indexes/scidocs/muvera/muvera_10240_R200_L600_a1.0/index_params.json"
  "results/indexes/scidocs/muvera/muvera_10240_R200_L600_a1.0/build_stats.json"
  "results/indexes/scidocs/svh_graph/svh_graph_R64_L128_a1.0/index_params.json"
  "results/indexes/scidocs/svh_graph/svh_graph_R64_L128_a1.0/build_stats.json"
  "results/indexes/scidocs/mvivf/mvivf_k0_l100_d0_nit5_s0_mpcc100/index_params.json"
  "results/indexes/scidocs/mvivf/mvivf_k0_l100_d0_nit5_s0_mpcc100/build_stats.json"
  "results/muvera_ablation/indices/nq/muvera/muvera_5120_R200_L600/index_params.json"
  "results/muvera_ablation/indices/nq/muvera/muvera_5120_R200_L600/build_stats.json"
  "results/muvera_ablation/indices/nq/muvera/muvera_10240_R200_L600/index_params.json"
  "results/muvera_ablation/indices/nq/muvera/muvera_10240_R200_L600/build_stats.json"
  "results/muvera_ablation/indices/nq/muvera/muvera_2560_R200_L600/index_params.json"
  "results/muvera_ablation/indices/nq/muvera/muvera_2560_R200_L600/build_stats.json"
)

moved=0
missing=0
for src in "${FILES[@]}"; do
  rel="${src#results/}"
  dst="$DEST_ROOT/$rel"
  if [[ ! -f "$src" ]]; then
    echo "MISSING: $src"
    missing=$((missing + 1))
    continue
  fi
  mkdir -p "$(dirname "$dst")"
  mv "$src" "$dst"
  echo "moved: $src -> $dst"
  moved=$((moved + 1))
done

echo
echo "Done. moved=$moved missing=$missing"
