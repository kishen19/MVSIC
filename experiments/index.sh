dataset=$1
maxsize=$2
iters=$3
rounds=$4


OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 bazel-bin/experiments/mvivf_benchmark -i /ssd2/laxman/multivector/${dataset}/${dataset}_points.pcs -o /ssd2/kishen/MVC/arguana/maxsize${maxsize}_iters${iters}/index_${maxsize}_${iters}.bin  -dist_func "Mips" -rounds ${rounds}