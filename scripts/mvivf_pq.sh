datasets=(arguana scidocs nq)

for dataset in ${datasets[@]}; do
  DATAPATH=/ssd2/laxman/multivector/${dataset}/${dataset}
  RESULTSPATH=/ssd2/kishen/MVC/grid_search/${dataset}

  mkdir -p ${RESULTSPATH}/indices
  mkdir -p ${RESULTSPATH}/results
  echo "Running mvivf_pq_benchmark for dataset: ${dataset}"
  bazel-bin/experiments/mvivf_pq_benchmark \
    -i ${DATAPATH}_points.pcs \
    -q ${DATAPATH}_queries.pcs \
    -gt ${DATAPATH}_chamfer_neighbors.gt \
    -index_dir ${RESULTSPATH}/indices \
    -results_dir ${RESULTSPATH}/results
done