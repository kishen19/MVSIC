dataset=$1

DATAPATH=/ssd2/laxman/multivector
RESULTSPATH=/ssd2/kishen/MVC

maxsizes=(500)
iters=(0 10)
ks=(10 100)

for maxsize in ${maxsizes[@]}; do
  for iter in ${iters[@]}; do
    for k in ${ks[@]}; do
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 \
      bazel-bin/experiments/mvivf_benchmark \
        -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
        -q  ${DATAPATH}/${dataset}/${dataset}_queries.pcs \
        -gt ${DATAPATH}/${dataset}/${dataset}_chamfer_neighbors.gt  \
        -index ${RESULTSPATH}/${dataset}/index_maxsize${maxsize}_iters${iter}.bin \
        -r ${RESULTSPATH}/${dataset}/maxsize${maxsize}_iters${iter}_k=${k}.csv \
        -k ${k} \
        -npl 1 -npr 256 -npmp 2 -npad 0
      echo ${dataset} ${maxsize} ${iter} ${k} done
    done
  done
done