dataset=$1
maxsize=$2
iters=$3
k=$4


OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 bazel-bin/experiments/mvivf_benchmark -i /ssd2/laxman/multivector/${dataset}/${dataset}_points.pcs -q /ssd2/laxman/multivector/${dataset}/${dataset}_queries.pcs -gt /ssd2/laxman/multivector/${dataset}/${dataset}_chamfer.gt -index /ssd2/kishen/MVC/arguana/maxsize${maxsize}_iters${iters}/index_${maxsize}_${iters}.bin -r /ssd2/kishen/MVC/${dataset}/maxsize${maxsize}_iters${iters}/k=${k}.csv  -dist_func "Mips" -k ${k}